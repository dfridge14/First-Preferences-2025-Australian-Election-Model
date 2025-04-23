<?php
/*
Plugin Name: Plotly Barplot JSON Importer
Description: Imports bar plot JSONs and attaches them to existing electorate posts.
Version: 1.0
*/

add_action('init', function () {
    if (!isset($_GET['import_barplots']) || $_GET['import_barplots'] !== 'true') return;

    $json_dir = WP_CONTENT_DIR . '/uploads/plotly-jsons-bars/';
    $json_files = glob($json_dir . '*_bar.json');// <-- Change the folder and match pattern if needed

    foreach ($json_files as $file_path) {
        $filename = basename($file_path);
        $electorate_name = preg_replace('/\.json$/', '', $filename);

        // Find existing post by title
        $post = get_page_by_title($electorate_name, OBJECT, 'electorate');

        if ($post) {
            update_field('FP_barplot', content_url('uploads/plotly-jsons-bars/' . $filename), $post->ID);
            echo "Updated {$electorate_name}<br>";
        } else {
            echo "Electorate {$electorate_name} not found!<br>";
        }
    }

    echo "Bar plot import completed.";
    exit;
});
