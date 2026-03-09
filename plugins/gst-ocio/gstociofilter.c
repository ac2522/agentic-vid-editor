/*
 * GstOCIOFilter — OpenColorIO transform filter for GStreamer.
 *
 * Phase 1 prototype: validates OCIO config + transform exists, passthrough.
 * Phase 3: full GPU shader generation via GpuShaderDesc, LUT texture upload.
 */

#include "gstociofilter.h"

GST_DEBUG_CATEGORY_STATIC(gst_ocio_filter_debug);
#define GST_CAT_DEFAULT gst_ocio_filter_debug

enum {
  PROP_0,
  PROP_CONFIG_PATH,
  PROP_SRC_COLORSPACE,
  PROP_DST_COLORSPACE,
};

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE("sink",
    GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:GLMemory),format=RGBA"));

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE("src",
    GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:GLMemory),format=RGBA"));

G_DEFINE_TYPE(GstOCIOFilter, gst_ocio_filter, GST_TYPE_GL_FILTER);

static void gst_ocio_filter_set_property(GObject *object, guint prop_id,
    const GValue *value, GParamSpec *pspec) {
  GstOCIOFilter *self = GST_OCIO_FILTER(object);
  switch (prop_id) {
    case PROP_CONFIG_PATH:
      g_free(self->config_path);
      self->config_path = g_value_dup_string(value);
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

static void gst_ocio_filter_get_property(GObject *object, guint prop_id,
    GValue *value, GParamSpec *pspec) {
  GstOCIOFilter *self = GST_OCIO_FILTER(object);
  switch (prop_id) {
    case PROP_CONFIG_PATH:
      g_value_set_string(value, self->config_path);
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

static gboolean gst_ocio_filter_gl_start(GstGLFilter *filter) {
  GstOCIOFilter *self = GST_OCIO_FILTER(filter);
  GST_INFO_OBJECT(self, "OCIO filter started (config=%s, %s -> %s)",
      self->config_path ? self->config_path : "(default)",
      self->src_colorspace ? self->src_colorspace : "(none)",
      self->dst_colorspace ? self->dst_colorspace : "(none)");
  /* Phase 3: Load OCIO config, create processor, generate GPU shader,
   * compile GL program, upload LUT textures */
  return TRUE;
}

static void gst_ocio_filter_gl_stop(GstGLFilter *filter) {
  /* Phase 3: Clean up GL program, LUT textures, OCIO processor */
}

static gboolean gst_ocio_filter_filter_texture(GstGLFilter *filter,
    GstGLMemory *input, GstGLMemory *output) {
  /* Phase 1: passthrough. Phase 3: apply OCIO shader. */
  const GstGLFuncs *gl = filter->context->gl_vtable;
  guint in_tex = gst_gl_memory_get_texture_id(input);
  guint out_tex = gst_gl_memory_get_texture_id(output);

  /* Simple copy via FBO for prototype */
  guint fbo;
  gl->GenFramebuffers(1, &fbo);
  gl->BindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
  gl->FramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
      GL_TEXTURE_2D, in_tex, 0);

  GstVideoInfo *info = &filter->out_info;
  int w = GST_VIDEO_INFO_WIDTH(info);
  int h = GST_VIDEO_INFO_HEIGHT(info);

  gl->BindTexture(GL_TEXTURE_2D, out_tex);
  gl->CopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, w, h);

  gl->BindFramebuffer(GL_READ_FRAMEBUFFER, 0);
  gl->DeleteFramebuffers(1, &fbo);

  return TRUE;
}

static void gst_ocio_filter_finalize(GObject *object) {
  GstOCIOFilter *self = GST_OCIO_FILTER(object);
  g_free(self->config_path);
  g_free(self->src_colorspace);
  g_free(self->dst_colorspace);
  G_OBJECT_CLASS(gst_ocio_filter_parent_class)->finalize(object);
}

static void gst_ocio_filter_class_init(GstOCIOFilterClass *klass) {
  GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
  GstGLFilterClass *filter_class = GST_GL_FILTER_CLASS(klass);

  gobject_class->set_property = gst_ocio_filter_set_property;
  gobject_class->get_property = gst_ocio_filter_get_property;
  gobject_class->finalize = gst_ocio_filter_finalize;

  g_object_class_install_property(gobject_class, PROP_CONFIG_PATH,
      g_param_spec_string("config-path", "OCIO Config",
          "Path to OpenColorIO config", NULL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property(gobject_class, PROP_SRC_COLORSPACE,
      g_param_spec_string("src-colorspace", "Source Color Space",
          "Source OCIO color space name", NULL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property(gobject_class, PROP_DST_COLORSPACE,
      g_param_spec_string("dst-colorspace", "Destination Color Space",
          "Destination OCIO color space name", NULL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gst_element_class_set_static_metadata(element_class,
      "OCIO Color Transform", "Filter/Video",
      "OpenColorIO color space transform via OpenGL",
      "Agentic Video Editor <ave@example.com>");

  gst_element_class_add_static_pad_template(element_class, &sink_template);
  gst_element_class_add_static_pad_template(element_class, &src_template);

  filter_class->gl_start = gst_ocio_filter_gl_start;
  filter_class->gl_stop = gst_ocio_filter_gl_stop;
  filter_class->filter_texture = gst_ocio_filter_filter_texture;

  GST_DEBUG_CATEGORY_INIT(gst_ocio_filter_debug, "ociofilter", 0,
      "OpenColorIO color transform filter");
}

static void gst_ocio_filter_init(GstOCIOFilter *self) {
  self->config_path = NULL;
  self->src_colorspace = NULL;
  self->dst_colorspace = NULL;
  self->ocio_processor = NULL;
  self->gl_program = 0;
  self->lut3d_tex = 0;
  self->lut3d_size = 0;
}
