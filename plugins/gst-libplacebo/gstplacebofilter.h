#ifndef __GST_PLACEBO_FILTER_H__
#define __GST_PLACEBO_FILTER_H__

#include <gst/gl/gstglfilter.h>
#include <libplacebo/opengl.h>
#include <libplacebo/renderer.h>
#include <libplacebo/log.h>

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

  /* Properties */
  gchar *lut_path;
};

G_END_DECLS

#endif /* __GST_PLACEBO_FILTER_H__ */
