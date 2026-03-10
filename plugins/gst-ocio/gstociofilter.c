/*
 * GstOCIOFilter — OpenColorIO transform filter for GStreamer.
 *
 * Uses a C++ helper (ocio_processor.cpp) to interface with OCIO's C++ API,
 * then uploads the baked 3D LUT as a GL texture and applies the OCIO-generated
 * GLSL shader to each frame via GstGLFilter.
 */

#include "gstociofilter.h"
#include "ocio_processor.h"
#include <gst/gl/gstglfuncs.h>
#include <string.h>

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

/* ---------- GLSL wrapper shader ---------- */

/*
 * We wrap the OCIO-generated shader function in a standard fragment shader.
 * The OCIO function processes vec4 color in-place.
 * The 3D LUT is bound as a sampler3D uniform.
 */

static const gchar *vert_shader_src =
    "#version 130\n"
    "in vec4 a_position;\n"
    "in vec2 a_texcoord;\n"
    "out vec2 v_texcoord;\n"
    "void main() {\n"
    "  gl_Position = a_position;\n"
    "  v_texcoord = a_texcoord;\n"
    "}\n";

/* Fragment shader template — %s is replaced with OCIO shader text + call */
static const gchar *frag_shader_template =
    "#version 130\n"
    "uniform sampler2D tex;\n"
    "%s\n"  /* OCIO-generated uniforms, textures, and function */
    "in vec2 v_texcoord;\n"
    "out vec4 fragColor;\n"
    "void main() {\n"
    "  vec4 col = texture(tex, v_texcoord);\n"
    "  %s(col);\n"  /* OCIO function call — modifies col in-place */
    "  fragColor = col;\n"
    "}\n";

/* ---------- Properties ---------- */

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

/* ---------- GL helpers ---------- */

static GLuint
compile_shader(const GstGLFuncs *gl, GLenum type, const gchar *source,
    GstOCIOFilter *self)
{
  GLuint shader = gl->CreateShader(type);
  if (!shader) {
    GST_ERROR_OBJECT(self, "glCreateShader failed");
    return 0;
  }

  gl->ShaderSource(shader, 1, &source, NULL);
  gl->CompileShader(shader);

  GLint status;
  gl->GetShaderiv(shader, GL_COMPILE_STATUS, &status);
  if (!status) {
    GLchar log[1024];
    gl->GetShaderInfoLog(shader, sizeof(log), NULL, log);
    GST_ERROR_OBJECT(self, "Shader compilation failed: %s", log);
    gl->DeleteShader(shader);
    return 0;
  }

  return shader;
}

static gboolean
build_gl_program(GstOCIOFilter *self, const GstGLFuncs *gl)
{
  const char *ocio_shader = ocio_processor_get_shader_text(
      (OcioProcessor *)self->ocio_processor);
  const char *ocio_func = ocio_processor_get_function_name(
      (OcioProcessor *)self->ocio_processor);

  if (!ocio_shader || !ocio_func) {
    GST_ERROR_OBJECT(self, "No OCIO shader text available");
    return FALSE;
  }

  /* Build fragment shader by inserting OCIO code */
  gchar *frag_src = g_strdup_printf(frag_shader_template,
      ocio_shader, ocio_func);

  GST_DEBUG_OBJECT(self, "Compiling OCIO fragment shader (%zu bytes)",
      strlen(frag_src));

  GLuint vert = compile_shader(gl, GL_VERTEX_SHADER, vert_shader_src, self);
  GLuint frag = compile_shader(gl, GL_FRAGMENT_SHADER, frag_src, self);
  g_free(frag_src);

  if (!vert || !frag) {
    if (vert) gl->DeleteShader(vert);
    if (frag) gl->DeleteShader(frag);
    return FALSE;
  }

  self->gl_program = gl->CreateProgram();
  gl->AttachShader(self->gl_program, vert);
  gl->AttachShader(self->gl_program, frag);
  gl->LinkProgram(self->gl_program);

  GLint status;
  gl->GetProgramiv(self->gl_program, GL_LINK_STATUS, &status);

  gl->DeleteShader(vert);
  gl->DeleteShader(frag);

  if (!status) {
    GLchar log[1024];
    gl->GetProgramInfoLog(self->gl_program, sizeof(log), NULL, log);
    GST_ERROR_OBJECT(self, "Shader link failed: %s", log);
    gl->DeleteProgram(self->gl_program);
    self->gl_program = 0;
    return FALSE;
  }

  GST_INFO_OBJECT(self, "OCIO GL shader program linked successfully");
  return TRUE;
}

static gboolean
upload_lut3d_texture(GstOCIOFilter *self, const GstGLFuncs *gl)
{
  int edge_len = ocio_processor_get_lut3d_size(
      (OcioProcessor *)self->ocio_processor);

  if (edge_len <= 0) {
    GST_DEBUG_OBJECT(self, "No 3D LUT to upload (edge_len=%d)", edge_len);
    return TRUE;  /* Not an error, some transforms don't use LUTs */
  }

  self->lut3d_size = edge_len;

  /* Allocate and fetch LUT data */
  gsize num_floats = (gsize)edge_len * edge_len * edge_len * 3;
  float *data = g_malloc(num_floats * sizeof(float));
  if (!data) {
    GST_ERROR_OBJECT(self, "Failed to allocate LUT data buffer");
    return FALSE;
  }

  ocio_processor_get_lut3d_data((OcioProcessor *)self->ocio_processor, data);

  /* Upload to GL 3D texture */
  gl->GenTextures(1, &self->lut3d_tex);
  gl->ActiveTexture(GL_TEXTURE1);
  gl->BindTexture(GL_TEXTURE_3D, self->lut3d_tex);
  gl->TexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  gl->TexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  gl->TexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  gl->TexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  gl->TexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  gl->TexImage3D(GL_TEXTURE_3D, 0, GL_RGB32F,
      edge_len, edge_len, edge_len, 0,
      GL_RGB, GL_FLOAT, data);

  gl->ActiveTexture(GL_TEXTURE0);
  g_free(data);

  GST_INFO_OBJECT(self, "Uploaded OCIO 3D LUT (%dx%dx%d) to GL texture %u",
      edge_len, edge_len, edge_len, self->lut3d_tex);

  return TRUE;
}

/* ---------- GL lifecycle ---------- */

static gboolean gst_ocio_filter_gl_start(GstGLFilter *filter) {
  GstOCIOFilter *self = GST_OCIO_FILTER(filter);
  const GstGLFuncs *gl = filter->context->gl_vtable;

  GST_INFO_OBJECT(self, "OCIO filter starting (config=%s, %s -> %s)",
      self->config_path ? self->config_path : "(env/$OCIO)",
      self->src_colorspace ? self->src_colorspace : "(none)",
      self->dst_colorspace ? self->dst_colorspace : "(none)");

  /* Validate required properties */
  if (!self->src_colorspace || !self->dst_colorspace) {
    GST_ERROR_OBJECT(self,
        "src-colorspace and dst-colorspace must both be set");
    return FALSE;
  }

  /* Create OCIO processor via C++ helper */
  const char *ocio_err = NULL;
  self->ocio_processor = ocio_processor_new(
      self->config_path, self->src_colorspace, self->dst_colorspace,
      &ocio_err);

  if (!self->ocio_processor) {
    GST_ERROR_OBJECT(self, "Failed to create OCIO processor: %s",
        ocio_err ? ocio_err : "unknown error");
    return FALSE;
  }

  /* Upload 3D LUT as GL texture */
  if (!upload_lut3d_texture(self, gl)) {
    GST_ERROR_OBJECT(self, "Failed to upload 3D LUT texture");
    ocio_processor_free((OcioProcessor *)self->ocio_processor);
    self->ocio_processor = NULL;
    return FALSE;
  }

  /* Compile and link OCIO GLSL shader program */
  if (!build_gl_program(self, gl)) {
    GST_ERROR_OBJECT(self, "Failed to build OCIO GL shader program");
    if (self->lut3d_tex) {
      gl->DeleteTextures(1, &self->lut3d_tex);
      self->lut3d_tex = 0;
    }
    ocio_processor_free((OcioProcessor *)self->ocio_processor);
    self->ocio_processor = NULL;
    return FALSE;
  }

  return TRUE;
}

static void gst_ocio_filter_gl_stop(GstGLFilter *filter) {
  GstOCIOFilter *self = GST_OCIO_FILTER(filter);
  const GstGLFuncs *gl = filter->context->gl_vtable;

  if (self->gl_program) {
    gl->DeleteProgram(self->gl_program);
    self->gl_program = 0;
  }

  if (self->lut3d_tex) {
    gl->DeleteTextures(1, &self->lut3d_tex);
    self->lut3d_tex = 0;
  }
  self->lut3d_size = 0;

  if (self->ocio_processor) {
    ocio_processor_free((OcioProcessor *)self->ocio_processor);
    self->ocio_processor = NULL;
  }
}

/* ---------- Filter ---------- */

static gboolean gst_ocio_filter_filter_texture(GstGLFilter *filter,
    GstGLMemory *input, GstGLMemory *output) {
  GstOCIOFilter *self = GST_OCIO_FILTER(filter);
  const GstGLFuncs *gl = filter->context->gl_vtable;

  if (!self->gl_program) {
    GST_ERROR_OBJECT(self, "No GL program available, cannot filter");
    return FALSE;
  }

  guint in_tex = gst_gl_memory_get_texture_id(input);
  guint out_tex = gst_gl_memory_get_texture_id(output);

  GstVideoInfo *info = &filter->out_info;
  int w = GST_VIDEO_INFO_WIDTH(info);
  int h = GST_VIDEO_INFO_HEIGHT(info);

  /* Bind output texture to FBO */
  guint fbo;
  gl->GenFramebuffers(1, &fbo);
  gl->BindFramebuffer(GL_FRAMEBUFFER, fbo);
  gl->FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
      GL_TEXTURE_2D, out_tex, 0);

  GLenum fbo_status = gl->CheckFramebufferStatus(GL_FRAMEBUFFER);
  if (fbo_status != GL_FRAMEBUFFER_COMPLETE) {
    GST_ERROR_OBJECT(self, "Framebuffer not complete: 0x%x", fbo_status);
    gl->BindFramebuffer(GL_FRAMEBUFFER, 0);
    gl->DeleteFramebuffers(1, &fbo);
    return FALSE;
  }

  gl->Viewport(0, 0, w, h);

  /* Use the OCIO shader program */
  gl->UseProgram(self->gl_program);

  /* Bind input texture to unit 0 */
  gl->ActiveTexture(GL_TEXTURE0);
  gl->BindTexture(GL_TEXTURE_2D, in_tex);
  GLint tex_loc = gl->GetUniformLocation(self->gl_program, "tex");
  if (tex_loc >= 0)
    gl->Uniform1i(tex_loc, 0);

  /* Bind 3D LUT texture to unit 1 (if present) */
  if (self->lut3d_tex) {
    gl->ActiveTexture(GL_TEXTURE1);
    gl->BindTexture(GL_TEXTURE_3D, self->lut3d_tex);

    /*
     * The OCIO shader expects the 3D LUT sampler with a specific name.
     * The default OCIO resource prefix is "ocio_" and the sampler name
     * comes from get3DTexture(). We bind it at texture unit 1.
     */
    GLint lut_loc = gl->GetUniformLocation(self->gl_program,
        "ocio_lut3d_0Sampler");
    if (lut_loc < 0) {
      /* Try alternative naming conventions */
      lut_loc = gl->GetUniformLocation(self->gl_program, "ocio_lut3d");
    }
    if (lut_loc >= 0)
      gl->Uniform1i(lut_loc, 1);

    gl->ActiveTexture(GL_TEXTURE0);
  }

  /* Draw fullscreen quad */
  static const GLfloat vertices[] = {
    /* position (x,y)    texcoord (s,t) */
    -1.0f, -1.0f,       0.0f, 0.0f,
     1.0f, -1.0f,       1.0f, 0.0f,
    -1.0f,  1.0f,       0.0f, 1.0f,
     1.0f,  1.0f,       1.0f, 1.0f,
  };

  GLuint vao, vbo;
  gl->GenVertexArrays(1, &vao);
  gl->BindVertexArray(vao);
  gl->GenBuffers(1, &vbo);
  gl->BindBuffer(GL_ARRAY_BUFFER, vbo);
  gl->BufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  GLint pos_loc = gl->GetAttribLocation(self->gl_program, "a_position");
  GLint tc_loc = gl->GetAttribLocation(self->gl_program, "a_texcoord");

  if (pos_loc >= 0) {
    gl->EnableVertexAttribArray(pos_loc);
    gl->VertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE,
        4 * sizeof(GLfloat), (void *)0);
  }
  if (tc_loc >= 0) {
    gl->EnableVertexAttribArray(tc_loc);
    gl->VertexAttribPointer(tc_loc, 2, GL_FLOAT, GL_FALSE,
        4 * sizeof(GLfloat), (void *)(2 * sizeof(GLfloat)));
  }

  gl->DrawArrays(GL_TRIANGLE_STRIP, 0, 4);

  /* Cleanup */
  if (pos_loc >= 0) gl->DisableVertexAttribArray(pos_loc);
  if (tc_loc >= 0) gl->DisableVertexAttribArray(tc_loc);
  gl->BindBuffer(GL_ARRAY_BUFFER, 0);
  gl->DeleteBuffers(1, &vbo);
  gl->BindVertexArray(0);
  gl->DeleteVertexArrays(1, &vao);

  gl->UseProgram(0);
  gl->BindFramebuffer(GL_FRAMEBUFFER, 0);
  gl->DeleteFramebuffers(1, &fbo);

  return TRUE;
}

/* ---------- Finalize ---------- */

static void gst_ocio_filter_finalize(GObject *object) {
  GstOCIOFilter *self = GST_OCIO_FILTER(object);
  g_free(self->config_path);
  g_free(self->src_colorspace);
  g_free(self->dst_colorspace);
  /* ocio_processor should already be freed in gl_stop */
  if (self->ocio_processor) {
    ocio_processor_free((OcioProcessor *)self->ocio_processor);
    self->ocio_processor = NULL;
  }
  G_OBJECT_CLASS(gst_ocio_filter_parent_class)->finalize(object);
}

/* ---------- Class/instance init ---------- */

static void gst_ocio_filter_class_init(GstOCIOFilterClass *klass) {
  GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
  GstGLFilterClass *filter_class = GST_GL_FILTER_CLASS(klass);

  gobject_class->set_property = gst_ocio_filter_set_property;
  gobject_class->get_property = gst_ocio_filter_get_property;
  gobject_class->finalize = gst_ocio_filter_finalize;

  g_object_class_install_property(gobject_class, PROP_CONFIG_PATH,
      g_param_spec_string("config-path", "OCIO Config",
          "Path to OpenColorIO config (or use $OCIO env var)", NULL,
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
      "OpenColorIO color space transform via GPU shader + 3D LUT",
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
