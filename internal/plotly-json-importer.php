<?php
/*
Plugin Name: Plotly JSON Importer
Description: Imports electorate posts and attaches Plotly JSON to each.
Version: 1.0
*/

add_action('init', function () {
    if (!isset($_GET['import_plotly']) || $_GET['import_plotly'] !== 'true') return;

    $json_dir = WP_CONTENT_DIR . '/uploads/plotly-jsons/';
    $json_files = glob($json_dir . '*_violin.json');

    foreach ($json_files as $file_path) {
        $filename = basename($file_path);
        $electorate_name = preg_replace('/_violin\.json$/', '', $filename);

        $post_id = wp_insert_post([
            'post_title' => $electorate_name,
            'post_name' => sanitize_title($electorate_name),
            'post_type' => 'electorate',
            'post_status' => 'publish'
        ]);

        if (!is_wp_error($post_id)) {
            update_field('FP_plots', content_url('uploads/plotly-jsons/' . $filename), $post_id);
        }
    }
https://chatgpt.com/c/67fdaa23-bbb0-8011-8217-d843b5d87b32
    echo "Electorate pages created.";
    exit;
});
