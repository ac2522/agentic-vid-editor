#include <gst/gst.h>
#include "gstociofilter.h"

static gboolean plugin_init(GstPlugin *plugin) {
  return gst_element_register(plugin, "ociofilter", GST_RANK_NONE,
      GST_TYPE_OCIO_FILTER);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    ocio,
    "OpenColorIO color transform filter",
    plugin_init,
    "0.1.0",
    "LGPL",
    "ave",
    "https://github.com/agentic-vid-editor"
)
