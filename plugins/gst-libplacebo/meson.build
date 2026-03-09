project('gst-libplacebo', 'c',
  version : '0.1.0',
  default_options : ['warning_level=2'])

gst_dep = dependency('gstreamer-1.0', version : '>= 1.28.1')
gst_gl_dep = dependency('gstreamer-gl-1.0', version : '>= 1.28.1')
gst_video_dep = dependency('gstreamer-video-1.0', version : '>= 1.28.1')
placebo_dep = dependency('libplacebo', version : '>= 7.349')

gstplacebo = shared_library('gstplacebo',
  'gstplacebofilter.c',
  'plugin.c',
  dependencies : [gst_dep, gst_gl_dep, gst_video_dep, placebo_dep],
  install : true,
  install_dir : join_paths(get_option('libdir'), 'gstreamer-1.0'),
)
