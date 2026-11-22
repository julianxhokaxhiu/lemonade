#pragma once

#include "lemon/backends/backend_descriptor.h"

namespace lemon {
namespace backends {
namespace strata {

inline const BackendDescriptor descriptor = {
    /*recipe*/ "strata",
    /*display_name*/ "Strata",
    /*binary*/ "strata",
    /*config_section*/ "strata",
    /*default_device*/ DEVICE_GPU,
    /*slot_policy*/ SlotPolicy::Standard,
    /*selectable_backend*/ true,
    /*uses_ctx_size*/ true,
    /*dynamic_models*/ false,
    /*options*/ {
        {"strata_backend", "--strata", "", "BACKEND",
         "Strata backend to use", "Strata Options"},
        {"strata_parallel", "--strata-parallel", 1, "SIZE",
         "Concurrent slots: Qwen 1-8, Hy4-Flash 1, Inkling 1-64", "Strata Options"},
        {"strata_drafting", "--strata-drafting", "auto", "MODE",
         "Drafting mode: auto, mtp, dflash2, off; shape engines use their built-in drafter",
         "Strata Options"},
        {"strata_args", "--strata-args", "", "ARGS",
         "Additional Strata engine arguments; model, context and serving mode are managed by Lemonade",
         "Strata Options"},
    },
    /*support*/ {
        {"cuda", BackendTier::Experimental, BackendFormat::Python, {"linux", "windows"},
         {{"nvidia_gpu", {"sm_60", "sm_61", "sm_70", "sm_75", "sm_86", "sm_89", "sm_120"}}},
         "NVIDIA Pascal and newer"},
        {"rocm", BackendTier::Experimental, BackendFormat::Python, {"linux", "windows"},
         {{"amd_gpu", {"gfx1030", "gfx1031", "gfx1032", "gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151",
                       "gfx1200", "gfx1201"}}},
         "AMD RX 6000/7000/9000, AI PRO R9700, Radeon 780M/890M, Strix Halo"},
        {"sycl", BackendTier::Experimental, BackendFormat::Python, {"linux"},
         {{"intel_gpu", {"arc"}}}, "Intel Arc A/B"},
        {"cuda", BackendTier::Experimental, BackendFormat::Python, {"linux"},
         {{"nvidia_gpu", {"sm_121"}}}, "NVIDIA GB10"},
        {"metal", BackendTier::Experimental, BackendFormat::Python, {"macos"},
         {{"metal", {}}}, "Apple Silicon"},
    },
    /*about*/ {
        {"cuda", {"Strata's Qwen, Hy4-Flash and Inkling engines with their bundled Python HTTP server.",
                  "Niko1221", "Niko1221", "MIT", "https://github.com/Niko1221/Strata"}},
        {"rocm", {"Strata's Qwen, Hy4-Flash and Inkling engines with their bundled Python HTTP server.",
                  "Niko1221", "Niko1221", "MIT", "https://github.com/Niko1221/Strata"}},
        {"sycl", {"Strata's MoE inference engine for Intel Arc.",
                  "Niko1221", "Niko1221", "MIT", "https://github.com/Niko1221/Strata"}},
        {"metal", {"Strata's Qwen, Hy4-Flash and Inkling engines for Apple Silicon.",
                   "Niko1221", "Niko1221", "MIT", "https://github.com/Niko1221/Strata"}},
    },
    /*supported_modes*/ {"chat"},
    /*required_checkpoints*/ {"main"},
    /*default_capabilities*/ {},
    /*web_display_name*/ "Strata",
    /*rocm_channels*/ {},
    /*exposes_prometheus_metrics*/ false,
    /*rocm_requires_cwsr_fix*/ false,
    /*version_policy*/ VersionPolicy::Exact,
    /*self_manages_downloads*/ false,
    /*takes_args*/ true,
    /*arg_variants*/ {},
    /*bin_variants*/ {"cuda", "rocm", "sycl", "metal"},
    /*config_extra*/ {{"parallel", 1}, {"drafting", "auto"}, {"runtime", "qwen"}},
    /*streams_model_from_storage*/ true,
};

}  // namespace strata
}  // namespace backends
}  // namespace lemon
