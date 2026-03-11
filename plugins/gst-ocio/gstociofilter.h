#ifndef __GST_OCIO_FILTER_H__
#define __GST_OCIO_FILTER_H__

#include <gst/gl/gstglfilter.h>

G_BEGIN_DECLS

#define GST_TYPE_OCIO_FILTER (gst_ocio_filter_get_type())
G_DECLARE_FINAL_TYPE(GstOCIOFilter, gst_ocio_filter, GST, OCIO_FILTER, GstGLFilter)

struct _GstOCIOFilter {
  GstGLFilter parent;

  /* Properties */
  gchar *config_path;
  gchar *src_colorspace;
  gchar *dst_colorspace;

  /* OCIO state (opaque, managed in .cpp helper) */
  gpointer ocio_processor;

  /* GL state */
  guint gl_program;
  guint lut3d_tex;
  gint lut3d_size;
  gchar *lut3d_sampler_name;  /* dynamically extracted from OCIO API */
};

G_END_DECLS

#endif /* __GST_OCIO_FILTER_H__ */
