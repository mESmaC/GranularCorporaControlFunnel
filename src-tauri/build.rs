fn main() {
    // tauri-build embeds its Windows resource (icon + version info + manifest)
    // into `bin` targets only. The manifest is what activates Common Controls
    // v6; without it the dialog plugin's `TaskDialogIndirect` import cannot be
    // resolved and *any* linked binary — including the unit-test harness —
    // dies at load with STATUS_ENTRYPOINT_NOT_FOUND. So we let tauri-build
    // handle icon/version only, and hand the manifest to the linker ourselves
    // for every linked target.
    let attrs = tauri_build::Attributes::new()
        .windows_attributes(tauri_build::WindowsAttributes::new_without_app_manifest());
    tauri_build::try_build(attrs).expect("failed to run tauri-build");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
        && std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc")
    {
        let manifest = std::env::current_dir()
            .expect("cwd")
            .join("windows-app-manifest.xml");
        println!("cargo:rerun-if-changed=windows-app-manifest.xml");
        println!("cargo:rustc-link-arg=/MANIFEST:EMBED");
        println!("cargo:rustc-link-arg=/MANIFESTINPUT:{}", manifest.display());
    }
}
