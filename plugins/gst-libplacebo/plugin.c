#include <gst/gst.h>
#include "gstplacebofilter.h"

static gboolean plugin_init(GstPlugin *plugin) {
  return gst_element_register(plugin, "placebofilter", GST_RANK_NONE,
      GST_TYPE_PLACEBO_FILTER);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    placebo,
    "libplacebo color processing filter (OpenGL backend)",
    plugin_init,
    "0.1.0",
    "LGPL",
    "ave",
    "https://github.com/agentic-vid-editor"
)
