project('gst-ocio', 'c', 'cpp',
  version : '0.1.0',
  default_options : ['warning_level=2', 'cpp_std=c++17'])

gst_dep = dependency('gstreamer-1.0', version : '>= 1.28.1')
gst_gl_dep = dependency('gstreamer-gl-1.0', version : '>= 1.28.1')
gst_video_dep = dependency('gstreamer-video-1.0', version : '>= 1.28.1')
ocio_dep = dependency('OpenColorIO', version : '>= 2.3')

gstocio = shared_library('gstocio',
  'gstociofilter.c',
  'ocio_processor.cpp',
  'plugin.c',
  dependencies : [gst_dep, gst_gl_dep, gst_video_dep, ocio_dep],
  install : true,
  install_dir : join_paths(get_option('libdir'), 'gstreamer-1.0'),
)
