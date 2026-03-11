/*
 * GstPlaceboFilter — libplacebo OpenGL backend filter for GStreamer.
 *
 * Supports .cube 3D LUT application via libplacebo renderer.
 * Uses OpenGL backend for GPU-accelerated color processing.
 *
 * Reference: FFmpeg vf_libplacebo.c (Vulkan-only, 1845 lines).
 * This element uses the OpenGL backend instead.
 */

#include "gstplacebofilter.h"
#include <gst/gl/gstglfuncs.h>

GST_DEBUG_CATEGORY_STATIC(gst_placebo_filter_debug);
#define GST_CAT_DEFAULT gst_placebo_filter_debug

enum {
  PROP_0,
  PROP_LUT_PATH,
  PROP_INTENSITY,
  PROP_SRC_COLORSPACE,
  PROP_DST_COLORSPACE,
};

#define DEFAULT_INTENSITY 1.0

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE("sink",
    GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:GLMemory),format=RGBA"));

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE("src",
    GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:GLMemory),format=RGBA"));

G_DEFINE_TYPE(GstPlaceboFilter, gst_placebo_filter, GST_TYPE_GL_FILTER);

/* ---------- LUT loading helper ---------- */

static gboolean
gst_placebo_filter_load_lut(GstPlaceboFilter *self)
{
  gchar *contents = NULL;
  gsize length = 0;
  GError *error = NULL;

  /* Free any previously loaded LUT */
  if (self->custom_lut) {
    pl_lut_free(&self->custom_lut);
    self->custom_lut = NULL;
  }

  if (!self->lut_path || self->lut_path[0] == '\0') {
    GST_DEBUG_OBJECT(self, "No LUT path set, skipping LUT load");
    return TRUE;
  }

  if (!g_file_get_contents(self->lut_path, &contents, &length, &error)) {
    GST_ERROR_OBJECT(self, "Failed to read LUT file '%s': %s",
        self->lut_path, error->message);
    g_error_free(error);
    return FALSE;
  }

  GST_INFO_OBJECT(self, "Parsing .cube LUT file '%s' (%zu bytes)",
      self->lut_path, length);

  self->custom_lut = pl_lut_parse_cube(self->pl_log, contents, length);
  g_free(contents);

  if (!self->custom_lut) {
    GST_ERROR_OBJECT(self, "Failed to parse .cube LUT file '%s'",
        self->lut_path);
    return FALSE;
  }

  GST_INFO_OBJECT(self, "Successfully loaded .cube LUT: %dx%dx%d",
      self->custom_lut->size[0],
      self->custom_lut->size[1],
      self->custom_lut->size[2]);

  return TRUE;
}

/* ---------- Properties ---------- */

static void gst_placebo_filter_set_property(GObject *object, guint prop_id,
    const GValue *value, GParamSpec *pspec) {
  GstPlaceboFilter *self = GST_PLACEBO_FILTER(object);
  switch (prop_id) {
    case PROP_LUT_PATH:
      g_free(self->lut_path);
      self->lut_path = g_value_dup_string(value);
      /* If the pipeline is already running, reload the LUT immediately */
      if (self->pl_log) {
        gst_placebo_filter_load_lut(self);
      }
      break;
    case PROP_INTENSITY:
      self->intensity = g_value_get_double(value);
      break;
    case PROP_SRC_COLORSPACE:
      g_free(self->src_colorspace);
      self->src_colorspace = g_value_dup_string(value);
      break;
    case PROP_DST_COLORSPACE:
      g_free(self->dst_colorspace);
      self->dst_colorspace = g_value_dup_string(value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

static void gst_placebo_filter_get_property(GObject *object, guint prop_id,
    GValue *value, GParamSpec *pspec) {
  GstPlaceboFilter *self = GST_PLACEBO_FILTER(object);
  switch (prop_id) {
    case PROP_LUT_PATH:
      g_value_set_string(value, self->lut_path);
      break;
    case PROP_INTENSITY:
      g_value_set_double(value, self->intensity);
      break;
    case PROP_SRC_COLORSPACE:
      g_value_set_string(value, self->src_colorspace);
      break;
    case PROP_DST_COLORSPACE:
      g_value_set_string(value, self->dst_colorspace);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

/* ---------- GL lifecycle ---------- */

static gboolean gst_placebo_filter_gl_start(GstGLFilter *filter) {
  GstPlaceboFilter *self = GST_PLACEBO_FILTER(filter);

  self->pl_log = pl_log_create(PL_API_VER, pl_log_params(
      .log_cb = NULL,
      .log_level = PL_LOG_WARN,
  ));

  self->pl_gl = pl_opengl_create(self->pl_log, pl_opengl_params(
      .allow_software = false,
  ));

  if (!self->pl_gl) {
    GST_ERROR_OBJECT(self, "Failed to create libplacebo OpenGL context");
    return FALSE;
  }

  self->pl_renderer = pl_renderer_create(self->pl_log, self->pl_gl->gpu);
  self->pl_dispatch = pl_dispatch_create(self->pl_log, self->pl_gl->gpu);

  GST_INFO_OBJECT(self, "libplacebo OpenGL backend initialized (GPU: %s)",
      self->pl_gl->gpu->glsl.version ? "yes" : "no");

  /* Load LUT if path was set before pipeline started */
  if (self->lut_path && self->lut_path[0] != '\0') {
    if (!gst_placebo_filter_load_lut(self)) {
      GST_WARNING_OBJECT(self, "LUT load failed during start, "
          "continuing without LUT");
    }
  }

  return TRUE;
}

static void gst_placebo_filter_gl_stop(GstGLFilter *filter) {
  GstPlaceboFilter *self = GST_PLACEBO_FILTER(filter);

  if (self->custom_lut) {
    pl_lut_free(&self->custom_lut);
    self->custom_lut = NULL;
  }

  pl_tex_destroy(self->pl_gl->gpu, &self->src_tex);
  pl_tex_destroy(self->pl_gl->gpu, &self->dst_tex);
  pl_tex_destroy(self->pl_gl->gpu, &self->blend_tex);
  pl_dispatch_destroy(&self->pl_dispatch);
  pl_renderer_destroy(&self->pl_renderer);
  pl_opengl_destroy(&self->pl_gl);
  pl_log_destroy(&self->pl_log);
}

/* ---------- Filter ---------- */

static gboolean gst_placebo_filter_filter_texture(GstGLFilter *filter,
    GstGLMemory *input, GstGLMemory *output) {
  GstPlaceboFilter *self = GST_PLACEBO_FILTER(filter);

  GstVideoInfo *in_info = &GST_GL_FILTER(self)->in_info;
  int w = GST_VIDEO_INFO_WIDTH(in_info);
  int h = GST_VIDEO_INFO_HEIGHT(in_info);

  /* Wrap GStreamer GL textures as libplacebo textures */
  struct pl_opengl_wrap_params src_wrap = {
      .width = w, .height = h,
      .texture = gst_gl_memory_get_texture_id(input),
      .target = GL_TEXTURE_2D,
      .iformat = GL_RGBA8,
  };
  struct pl_opengl_wrap_params dst_wrap = {
      .width = w, .height = h,
      .texture = gst_gl_memory_get_texture_id(output),
      .target = GL_TEXTURE_2D,
      .iformat = GL_RGBA8,
  };

  pl_tex src = pl_opengl_wrap(self->pl_gl->gpu, &src_wrap);
  pl_tex dst = pl_opengl_wrap(self->pl_gl->gpu, &dst_wrap);

  if (!src || !dst) {
    GST_ERROR_OBJECT(self, "Failed to wrap GL textures");
    if (src) pl_tex_destroy(self->pl_gl->gpu, &src);
    if (dst) pl_tex_destroy(self->pl_gl->gpu, &dst);
    return FALSE;
  }

  gdouble intensity = self->intensity;
  gboolean need_blend = (intensity > 0.0 && intensity < 1.0
                          && self->custom_lut != NULL);

  /*
   * Fast path: intensity <= 0.0 or no LUT means passthrough.
   * Render source to output without any LUT applied.
   */
  if (intensity <= 0.0 || !self->custom_lut) {
    struct pl_frame img = {
        .num_planes = 1,
        .planes = {{ .texture = src,
                      .components = 4,
                      .component_mapping = {0, 1, 2, 3} }},
        .repr = pl_color_repr_sdtv,
        .color = pl_color_space_srgb,
    };
    struct pl_frame target = {
        .num_planes = 1,
        .planes = {{ .texture = dst,
                      .components = 4,
                      .component_mapping = {0, 1, 2, 3} }},
        .repr = pl_color_repr_sdtv,
        .color = pl_color_space_srgb,
    };
    struct pl_render_params params = pl_render_default_params;

    GST_TRACE_OBJECT(self, "Intensity %.2f — passthrough (no LUT)", intensity);

    gboolean ok = pl_render_image(self->pl_renderer, &img, &target, &params);
    if (!ok)
      GST_ERROR_OBJECT(self, "pl_render_image() failed (passthrough)");
    pl_tex_destroy(self->pl_gl->gpu, &src);
    pl_tex_destroy(self->pl_gl->gpu, &dst);
    return ok;
  }

  /*
   * For blending (0 < intensity < 1), we need a temporary texture to hold
   * the LUT-applied result, then mix original and LUT result together.
   * For full intensity (>= 1.0), render LUT directly to the output.
   */
  pl_tex lut_target = dst;  /* default: render LUT result straight to output */

  if (need_blend) {
    /* Ensure the temporary blend texture exists with the right dimensions */
    pl_fmt fmt = src->params.format;
    gboolean ok = pl_tex_recreate(self->pl_gl->gpu, &self->blend_tex,
        pl_tex_params(
            .w = w, .h = h,
            .format = fmt,
            .sampleable = true,
            .renderable = true,
        ));
    if (!ok) {
      GST_ERROR_OBJECT(self, "Failed to create blend temp texture");
      pl_tex_destroy(self->pl_gl->gpu, &src);
      pl_tex_destroy(self->pl_gl->gpu, &dst);
      return FALSE;
    }
    lut_target = self->blend_tex;  /* render LUT result to temp texture */
  }

  /* Set up source frame */
  struct pl_frame img = {
      .num_planes = 1,
      .planes = {{ .texture = src,
                    .components = 4,
                    .component_mapping = {0, 1, 2, 3} }},
      .repr = pl_color_repr_sdtv,
      .color = pl_color_space_srgb,
  };

  /* Set up target frame — goes to blend_tex when blending, dst otherwise */
  struct pl_frame target = {
      .num_planes = 1,
      .planes = {{ .texture = lut_target,
                    .components = 4,
                    .component_mapping = {0, 1, 2, 3} }},
      .repr = pl_color_repr_sdtv,
      .color = pl_color_space_srgb,
  };

  /* Configure render params with LUT */
  struct pl_render_params params = pl_render_default_params;
  params.lut = self->custom_lut;
  params.lut_type = PL_LUT_NORMALIZED;
  GST_TRACE_OBJECT(self, "Applying .cube LUT to frame (intensity=%.2f)",
      intensity);

  gboolean ok = pl_render_image(self->pl_renderer, &img, &target, &params);
  if (!ok) {
    GST_ERROR_OBJECT(self, "pl_render_image() failed (LUT pass)");
    pl_tex_destroy(self->pl_gl->gpu, &src);
    pl_tex_destroy(self->pl_gl->gpu, &dst);
    return FALSE;
  }

  /*
   * Blending pass: mix(original, lut_applied, intensity) via custom shader.
   * Only executed when 0 < intensity < 1.  Uses pl_dispatch + pl_shader_custom
   * to run a trivial GLSL mix shader that samples the original input and the
   * LUT-processed temporary texture, then writes the blended result to the
   * output texture.
   */
  if (need_blend) {
    pl_shader sh = pl_dispatch_begin(self->pl_dispatch);

    /* Build the mix shader using pl_shader_custom with texture descriptors */
    struct pl_shader_desc descs[2] = {
        {
            .desc = {
                .name = "orig_tex",
                .type = PL_DESC_SAMPLED_TEX,
            },
            .binding = {
                .object = src,
                .sample_mode = PL_TEX_SAMPLE_LINEAR,
            },
        },
        {
            .desc = {
                .name = "lut_tex",
                .type = PL_DESC_SAMPLED_TEX,
            },
            .binding = {
                .object = self->blend_tex,
                .sample_mode = PL_TEX_SAMPLE_LINEAR,
            },
        },
    };

    float fintensity = (float) intensity;
    struct pl_shader_var vars[1] = {
        {
            .var = {
                .name = "intensity",
                .type = PL_VAR_FLOAT,
                .dim_v = 1,
                .dim_m = 1,
            },
            .data = &fintensity,
        },
    };

    char body[512];
    g_snprintf(body, sizeof(body),
        "vec2 uv = gl_FragCoord.xy / vec2(float(%d), float(%d));\n"
        "vec4 orig = textureLod(orig_tex, uv, 0.0);\n"
        "vec4 lut  = textureLod(lut_tex,  uv, 0.0);\n"
        "color = mix(orig, lut, intensity);\n",
        w, h);

    struct pl_custom_shader custom = {
        .description = "LUT intensity blend",
        .body = body,
        .input = PL_SHADER_SIG_NONE,
        .output = PL_SHADER_SIG_COLOR,
        .descriptors = descs,
        .num_descriptors = 2,
        .variables = vars,
        .num_variables = 1,
    };

    if (!pl_shader_custom(sh, &custom)) {
      GST_ERROR_OBJECT(self, "pl_shader_custom() failed for blend pass");
      pl_dispatch_abort(self->pl_dispatch, &sh);
      pl_tex_destroy(self->pl_gl->gpu, &src);
      pl_tex_destroy(self->pl_gl->gpu, &dst);
      return FALSE;
    }

    struct pl_dispatch_params dparams = {
        .shader = &sh,
        .target = dst,
    };

    ok = pl_dispatch_finish(self->pl_dispatch, &dparams);
    if (!ok) {
      GST_ERROR_OBJECT(self, "pl_dispatch_finish() failed for blend pass");
      pl_tex_destroy(self->pl_gl->gpu, &src);
      pl_tex_destroy(self->pl_gl->gpu, &dst);
      return FALSE;
    }

    GST_TRACE_OBJECT(self, "Intensity blend pass completed (%.2f)", intensity);
  }

  pl_tex_destroy(self->pl_gl->gpu, &src);
  pl_tex_destroy(self->pl_gl->gpu, &dst);
  return TRUE;
}

/* ---------- Finalize ---------- */

static void gst_placebo_filter_finalize(GObject *object) {
  GstPlaceboFilter *self = GST_PLACEBO_FILTER(object);
  g_free(self->lut_path);
  g_free(self->src_colorspace);
  g_free(self->dst_colorspace);
  /* custom_lut should already be freed in gl_stop, but guard just in case */
  if (self->custom_lut) {
    pl_lut_free(&self->custom_lut);
    self->custom_lut = NULL;
  }
  G_OBJECT_CLASS(gst_placebo_filter_parent_class)->finalize(object);
}

/* ---------- Class/instance init ---------- */

static void gst_placebo_filter_class_init(GstPlaceboFilterClass *klass) {
  GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
  GstGLFilterClass *filter_class = GST_GL_FILTER_CLASS(klass);

  gobject_class->set_property = gst_placebo_filter_set_property;
  gobject_class->get_property = gst_placebo_filter_get_property;
  gobject_class->finalize = gst_placebo_filter_finalize;

  g_object_class_install_property(gobject_class, PROP_LUT_PATH,
      g_param_spec_string("lut-path", "LUT Path",
          "Path to a .cube 3D LUT file", NULL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property(gobject_class, PROP_INTENSITY,
      g_param_spec_double("intensity", "LUT Intensity",
          "Blend strength of LUT application (0.0=bypass, 1.0=full)",
          0.0, 1.0, DEFAULT_INTENSITY,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property(gobject_class, PROP_SRC_COLORSPACE,
      g_param_spec_string("src-colorspace", "Source Color Space",
          "Source color space name (for metadata/future use)", NULL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property(gobject_class, PROP_DST_COLORSPACE,
      g_param_spec_string("dst-colorspace", "Destination Color Space",
          "Destination color space name (for metadata/future use)", NULL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gst_element_class_set_static_metadata(element_class,
      "libplacebo Filter", "Filter/Video",
      "Color processing via libplacebo OpenGL backend (LUT, tone mapping)",
      "Agentic Video Editor <ave@example.com>");

  gst_element_class_add_static_pad_template(element_class, &sink_template);
  gst_element_class_add_static_pad_template(element_class, &src_template);

  filter_class->gl_start = gst_placebo_filter_gl_start;
  filter_class->gl_stop = gst_placebo_filter_gl_stop;
  filter_class->filter_texture = gst_placebo_filter_filter_texture;

  GST_DEBUG_CATEGORY_INIT(gst_placebo_filter_debug, "placebofilter", 0,
      "libplacebo color filter");
}

static void gst_placebo_filter_init(GstPlaceboFilter *self) {
  self->lut_path = NULL;
  self->intensity = DEFAULT_INTENSITY;
  self->src_colorspace = NULL;
  self->dst_colorspace = NULL;
  self->custom_lut = NULL;
  self->pl_log = NULL;
  self->pl_gl = NULL;
  self->pl_renderer = NULL;
  self->pl_dispatch = NULL;
  self->src_tex = NULL;
  self->dst_tex = NULL;
  self->blend_tex = NULL;
}
