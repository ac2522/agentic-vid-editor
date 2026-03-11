#ifndef __GST_PLACEBO_FILTER_H__
#define __GST_PLACEBO_FILTER_H__

#include <gst/gl/gstglfilter.h>
#include <libplacebo/opengl.h>
#include <libplacebo/renderer.h>
#include <libplacebo/log.h>
#include <libplacebo/shaders/lut.h>
#include <libplacebo/shaders/custom.h>
#include <libplacebo/dispatch.h>

G_BEGIN_DECLS

#define GST_TYPE_PLACEBO_FILTER (gst_placebo_filter_get_type())
G_DECLARE_FINAL_TYPE(GstPlaceboFilter, gst_placebo_filter, GST, PLACEBO_FILTER, GstGLFilter)

struct _GstPlaceboFilter {
  GstGLFilter parent;

  /* libplacebo state */
  pl_log pl_log;
  pl_opengl pl_gl;
  pl_renderer pl_renderer;
  pl_tex src_tex;
  pl_tex dst_tex;
  pl_dispatch pl_dispatch;
  pl_tex blend_tex;  /* temporary texture for intensity blending */

  /* Properties */
  gchar *lut_path;
  gdouble intensity;
  gchar *src_colorspace;
  gchar *dst_colorspace;

  /* Parsed LUT data */
  struct pl_custom_lut *custom_lut;
};

G_END_DECLS

#endif /* __GST_PLACEBO_FILTER_H__ */
