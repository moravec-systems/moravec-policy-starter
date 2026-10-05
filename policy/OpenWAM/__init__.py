# moravec_policy.setup_policy_server imports `moravec_policy.policy.OpenWAM.model.Model`
# directly, so this file only needs to mark the directory as a package.
# Keep it import-free: pulling .model here would drag torch/OpenWAM into any
# code that merely walks moravec_policy.policy.*.
