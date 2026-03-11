/*
 * ocio_processor.cpp — OpenColorIO C++ wrapper with C-compatible interface.
 *
 * Uses OCIO 2.x API to:
 *   1. Load a config from file or $OCIO environment variable
 *   2. Create a Processor for src -> dst color space transform
 *   3. Extract GPU shader info (GLSL text + baked 3D LUT data)
 *
 * The baked 3D LUT approach ("legacy" shader mode) produces a single 3D LUT
 * texture that can be uploaded as a GL 3D texture and sampled in the shader.
 */

#include "ocio_processor.h"

#include <OpenColorIO/OpenColorIO.h>
#include <cstring>
#include <string>
#include <vector>

namespace OCIO = OCIO_NAMESPACE;

/* Thread-local buffer for error messages from failed ocio_processor_new calls.
 * Needed because the OcioProcessor struct is deleted on failure, so the
 * caller cannot retrieve error_msg via ocio_processor_get_error(NULL). */
static thread_local std::string g_last_error;

struct OcioProcessor {
    OCIO::ConstConfigRcPtr config;
    OCIO::ConstProcessorRcPtr processor;
    OCIO::ConstGPUProcessorRcPtr gpu_processor;
    OCIO::GpuShaderDescRcPtr shader_desc;

    /* Cached shader text (owned by shader_desc, but we keep the pointer) */
    std::string shader_text;
    std::string function_name;

    /* Baked 3D LUT */
    int lut3d_edge_len;
    std::vector<float> lut3d_data;

    /* Cached 3D texture sampler names from GpuShaderDesc */
    std::vector<std::string> lut3d_sampler_names;

    /* Error reporting */
    std::string error_msg;
};

extern "C" {

static void set_error(const char** error_out, const std::string& msg) {
    g_last_error = msg;
    if (error_out) {
        *error_out = g_last_error.c_str();
    }
}

OcioProcessor* ocio_processor_new(const char* config_path,
                                  const char* src_cs,
                                  const char* dst_cs,
                                  const char** error_out)
{
    if (error_out) *error_out = nullptr;

    if (!src_cs || !dst_cs) {
        set_error(error_out, "src_cs and dst_cs must not be NULL");
        return nullptr;
    }

    OcioProcessor* proc = new (std::nothrow) OcioProcessor();
    if (!proc) {
        return nullptr;
    }

    try {
        /* Load config */
        if (config_path && config_path[0] != '\0') {
            proc->config = OCIO::Config::CreateFromFile(config_path);
        } else {
            proc->config = OCIO::Config::CreateFromEnv();
        }

        if (!proc->config) {
            set_error(error_out, "Failed to load OCIO config");
            delete proc;
            return nullptr;
        }

        /* Validate that source and destination color spaces exist */
        if (!proc->config->hasColorSpace(src_cs)) {
            std::string msg = "Color space '";
            msg += src_cs;
            msg += "' not found in OCIO config";
            set_error(error_out, msg);
            delete proc;
            return nullptr;
        }
        if (!proc->config->hasColorSpace(dst_cs)) {
            std::string msg = "Color space '";
            msg += dst_cs;
            msg += "' not found in OCIO config";
            set_error(error_out, msg);
            delete proc;
            return nullptr;
        }

        /* Create processor for color space transform */
        proc->processor = proc->config->getProcessor(src_cs, dst_cs);
        if (!proc->processor) {
            std::string msg = "Failed to create OCIO processor for transform: ";
            msg += src_cs;
            msg += " -> ";
            msg += dst_cs;
            set_error(error_out, msg);
            delete proc;
            return nullptr;
        }

        /* Get GPU processor (optimized for GPU execution) */
        proc->gpu_processor = proc->processor->getDefaultGPUProcessor();
        if (!proc->gpu_processor) {
            set_error(error_out, "Failed to get GPU processor");
            delete proc;
            return nullptr;
        }

        /* Create shader description and extract GPU info */
        proc->shader_desc = OCIO::GpuShaderDesc::CreateShaderDesc();
        proc->shader_desc->setLanguage(OCIO::GPU_LANGUAGE_GLSL_1_3);
        proc->shader_desc->setFunctionName("OCIOColor");
        proc->shader_desc->setResourcePrefix("ocio_");

        proc->gpu_processor->extractGpuShaderInfo(proc->shader_desc);

        /* Cache shader text */
        const char* text = proc->shader_desc->getShaderText();
        if (text) {
            proc->shader_text = text;
        }
        proc->function_name = "OCIOColor";

        /* Validate shader text is non-empty */
        if (proc->shader_text.empty()) {
            set_error(error_out,
                "OCIO produced empty shader text for transform");
            delete proc;
            return nullptr;
        }

        /* Extract 3D LUT data if present */
        unsigned num_3d_textures = proc->shader_desc->getNum3DTextures();
        if (num_3d_textures > 0) {
            /* Cache sampler names for all 3D textures */
            for (unsigned i = 0; i < num_3d_textures; ++i) {
                const char* t_name = nullptr;
                const char* s_name = nullptr;
                unsigned el = 0;
                OCIO::Interpolation ip = OCIO::INTERP_LINEAR;
                proc->shader_desc->get3DTexture(i, t_name, s_name, el, ip);
                proc->lut3d_sampler_names.push_back(
                    s_name ? std::string(s_name) : std::string());
            }

            const char* tex_name = nullptr;
            const char* sampler_name = nullptr;
            unsigned edgelen = 0;
            OCIO::Interpolation interp = OCIO::INTERP_LINEAR;

            proc->shader_desc->get3DTexture(0, tex_name, sampler_name,
                                            edgelen, interp);

            proc->lut3d_edge_len = static_cast<int>(edgelen);

            /* Validate LUT size */
            if (edgelen == 0) {
                set_error(error_out,
                    "OCIO reported 3D LUT with zero edge length");
                delete proc;
                return nullptr;
            }

            /* Get LUT data: edgelen^3 texels, 3 floats (RGB) each */
            const float* values = nullptr;
            proc->shader_desc->get3DTextureValues(0, values);

            if (values && edgelen > 0) {
                size_t num_floats = static_cast<size_t>(edgelen) *
                                    static_cast<size_t>(edgelen) *
                                    static_cast<size_t>(edgelen) * 3;
                proc->lut3d_data.assign(values, values + num_floats);
            }

            if (proc->lut3d_data.empty()) {
                set_error(error_out,
                    "OCIO 3D LUT has valid size but no data");
                delete proc;
                return nullptr;
            }
        } else {
            proc->lut3d_edge_len = 0;
        }

        return proc;

    } catch (const OCIO::Exception& e) {
        std::string msg = "OCIO exception: ";
        msg += e.what();
        set_error(error_out, msg);
        delete proc;
        return nullptr;
    } catch (const std::exception& e) {
        std::string msg = "C++ exception: ";
        msg += e.what();
        set_error(error_out, msg);
        delete proc;
        return nullptr;
    }
}

void ocio_processor_free(OcioProcessor* proc)
{
    delete proc;
}

int ocio_processor_get_lut3d_size(OcioProcessor* proc)
{
    if (!proc) return 0;
    return proc->lut3d_edge_len;
}

void ocio_processor_get_lut3d_data(OcioProcessor* proc, float* data)
{
    if (!proc || !data || proc->lut3d_data.empty()) return;
    std::memcpy(data, proc->lut3d_data.data(),
                proc->lut3d_data.size() * sizeof(float));
}

const char* ocio_processor_get_shader_text(OcioProcessor* proc)
{
    if (!proc || proc->shader_text.empty()) return nullptr;
    return proc->shader_text.c_str();
}

const char* ocio_processor_get_function_name(OcioProcessor* proc)
{
    if (!proc || proc->function_name.empty()) return nullptr;
    return proc->function_name.c_str();
}

const char* ocio_processor_get_error(OcioProcessor* proc)
{
    if (!proc || proc->error_msg.empty()) return nullptr;
    return proc->error_msg.c_str();
}

const char* ocio_processor_get_texture_name(OcioProcessor* proc, int index)
{
    if (!proc) return nullptr;
    if (index < 0 ||
        static_cast<size_t>(index) >= proc->lut3d_sampler_names.size()) {
        return nullptr;
    }
    const std::string& name = proc->lut3d_sampler_names[static_cast<size_t>(index)];
    if (name.empty()) return nullptr;
    return name.c_str();
}

int ocio_processor_validate_colorspace(const char* config_path,
                                       const char* colorspace_name)
{
    if (!colorspace_name) return -1;

    try {
        OCIO::ConstConfigRcPtr config;
        if (config_path && config_path[0] != '\0') {
            config = OCIO::Config::CreateFromFile(config_path);
        } else {
            config = OCIO::Config::CreateFromEnv();
        }

        if (!config) return -1;

        return config->hasColorSpace(colorspace_name) ? 1 : 0;
    } catch (const OCIO::Exception&) {
        return -1;
    } catch (const std::exception&) {
        return -1;
    }
}

} /* extern "C" */
