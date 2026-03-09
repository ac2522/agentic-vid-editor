/*
 * GstPlaceboFilter — libplacebo OpenGL backend filter for GStreamer.
 *
 * Phase 1 prototype: passthrough + basic 3D LUT application.
 * Phase 3: full color management, tone mapping, HDR, scaling.
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
};

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE("sink",
    GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:GLMemory),format=RGBA"));

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE("src",
    GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:GLMemory),format=RGBA"));

G_DEFINE_TYPE(GstPlaceboFilter, gst_placebo_filter, GST_TYPE_GL_FILTER);

static void gst_placebo_filter_set_property(GObject *object, guint prop_id,
    const GValue *value, GParamSpec *pspec) {
  GstPlaceboFilter *self = GST_PLACEBO_FILTER(object);
  switch (prop_id) {
    case PROP_LUT_PATH:
      g_free(self->lut_path);
      self->lut_path = g_value_dup_string(value);
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
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

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

  GST_INFO_OBJECT(self, "libplacebo OpenGL backend initialized (GPU: %s)",
      self->pl_gl->gpu->glsl.version ? "yes" : "no");

  return TRUE;
}

static void gst_placebo_filter_gl_stop(GstGLFilter *filter) {
  GstPlaceboFilter *self = GST_PLACEBO_FILTER(filter);

  pl_tex_destroy(self->pl_gl->gpu, &self->src_tex);
  pl_tex_destroy(self->pl_gl->gpu, &self->dst_tex);
  pl_renderer_destroy(&self->pl_renderer);
  pl_opengl_destroy(&self->pl_gl);
  pl_log_destroy(&self->pl_log);
}

static gboolean gst_placebo_filter_filter_texture(GstGLFilter *filter,
    GstGLMemory *input, GstGLMemory *output) {
  GstPlaceboFilter *self = GST_PLACEBO_FILTER(filter);

  /* Phase 1: passthrough — copy input to output via libplacebo renderer.
   * Phase 3 will add: LUT application, tone mapping, color management. */

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

  /* Render: passthrough for now */
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

  gboolean ok = pl_render_image(self->pl_renderer, &img, &target, &params);

  pl_tex_destroy(self->pl_gl->gpu, &src);
  pl_tex_destroy(self->pl_gl->gpu, &dst);

  return ok;
}

static void gst_placebo_filter_finalize(GObject *object) {
  GstPlaceboFilter *self = GST_PLACEBO_FILTER(object);
  g_free(self->lut_path);
  G_OBJECT_CLASS(gst_placebo_filter_parent_class)->finalize(object);
}

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

  gst_element_class_set_static_metadata(element_class,
      "libplacebo Filter", "Filter/Video",
      "Color processing via libplacebo OpenGL backend",
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
  self->pl_log = NULL;
  self->pl_gl = NULL;
  self->pl_renderer = NULL;
  self->src_tex = NULL;
  self->dst_tex = NULL;
}
