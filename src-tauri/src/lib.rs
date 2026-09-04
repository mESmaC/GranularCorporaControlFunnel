pub mod classifier;
pub mod commands;
pub mod db;
pub mod error;
pub mod export;
pub mod formats;
pub mod jobs;
pub mod receipt;
pub mod rng;
pub mod sample;
pub mod scan;
pub mod session;

use commands::AppState;
use std::path::PathBuf;
use std::sync::Arc;
use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).try_init();

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let db_path = match std::env::var("GCCF_DB") {
                Ok(p) if !p.trim().is_empty() => PathBuf::from(p),
                _ => app.path().app_data_dir()?.join("gccf.sqlite3"),
            };
            log::info!("catalogue: {}", db_path.display());
            let db = db::Db::open(&db_path).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
            app.manage(AppState {
                db: Arc::new(db),
                jobs: Arc::new(jobs::Jobs::new()),
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::app_info,
            commands::get_settings,
            commands::set_setting,
            commands::list_jobs,
            commands::cancel_job,
            commands::list_roots,
            commands::add_root,
            commands::rescan_root,
            commands::remove_root,
            commands::root_format_stats,
            commands::list_sessions,
            commands::get_session,
            commands::create_session,
            commands::delete_session,
            commands::rename_session,
            commands::next_sample,
            commands::get_sample,
            commands::decide_sample,
            commands::set_sample_tag,
            commands::list_samples,
            commands::session_counts,
            commands::list_tags,
            commands::create_tag,
            commands::delete_tag,
            commands::rename_tag,
            commands::set_tag_hotkey,
            commands::reveal_file,
            commands::open_file,
            commands::reveal_path,
            commands::start_classifier_run,
            commands::list_classifier_runs,
            commands::apply_run_predictions,
            commands::delete_classifier_run,
            commands::list_models,
            commands::delete_model,
            commands::train_model,
            commands::eval_report,
            commands::export_training_data,
            commands::receipt_preview,
            commands::export_receipt,
        ])
        .run(tauri::generate_context!())
        .expect("error while running GCCF");
}
