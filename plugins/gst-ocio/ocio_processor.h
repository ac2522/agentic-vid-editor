/*
 * ocio_processor.h — C-compatible interface to OpenColorIO C++ API.
 *
 * Wraps OCIO Config, Processor, and GpuShaderDesc into an opaque handle
 * that can be used from pure C GStreamer plugin code.
 */

#ifndef __OCIO_PROCESSOR_H__
#define __OCIO_PROCESSOR_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OcioProcessor OcioProcessor;

/*
 * Create an OCIO processor for a color space transform.
 *
 * @param config_path  Path to OCIO config file, or NULL to use $OCIO env var.
 * @param src_cs       Source color space name (must not be NULL).
 * @param dst_cs       Destination color space name (must not be NULL).
 * @return             Opaque processor handle, or NULL on failure.
 */
OcioProcessor* ocio_processor_new(const char* config_path,
                                  const char* src_cs,
                                  const char* dst_cs);

/*
 * Free an OCIO processor and all associated resources.
 */
void ocio_processor_free(OcioProcessor* proc);

/*
 * Get the edge length of the baked 3D LUT.
 * The full LUT has size^3 * 3 float values.
 *
 * @return Edge length (e.g. 65), or 0 on error.
 */
int ocio_processor_get_lut3d_size(OcioProcessor* proc);

/*
 * Copy the baked 3D LUT data into the provided buffer.
 * Buffer must hold at least (size^3 * 3) floats.
 *
 * @param data  Output buffer for LUT values (RGB, float).
 */
void ocio_processor_get_lut3d_data(OcioProcessor* proc, float* data);

/*
 * Get the GLSL shader text for the OCIO transform.
 * The returned string is owned by the processor and valid until it is freed.
 *
 * @return GLSL shader text, or NULL on error.
 */
const char* ocio_processor_get_shader_text(OcioProcessor* proc);

/*
 * Get the GLSL function name used in the shader.
 * Useful for calling the OCIO function from a wrapper fragment shader.
 *
 * @return Function name string, or NULL on error.
 */
const char* ocio_processor_get_function_name(OcioProcessor* proc);

/*
 * Get the last error message, or NULL if no error occurred.
 */
const char* ocio_processor_get_error(OcioProcessor* proc);

#ifdef __cplusplus
}
#endif

#endif /* __OCIO_PROCESSOR_H__ */
