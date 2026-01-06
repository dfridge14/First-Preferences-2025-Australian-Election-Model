<?php
/**
 * The Template for displaying all single posts.
 *
 * @package GeneratePress
 */

if ( ! defined( 'ABSPATH' ) ) {
	exit; // Exit if accessed directly.
}

get_header(); ?>

	<div <?php generate_do_attr( 'content' ); ?>>
		<main <?php generate_do_attr( 'main' ); ?>>
			<?php
			/**
			 * generate_before_main_content hook.
			 *
			 * @since 0.1
			 */
			do_action( 'generate_before_main_content' );

			if ( generate_has_default_loop() ) {
				while ( have_posts() ) :

					the_post();

					// Display the post content.
					generate_do_template_part( 'single' );

					// Display the Plotly chart below the content
					$plotly_json = get_field('FP_plots'); // Fetch the Plotly JSON path from ACF field
					if ($plotly_json) {
						// Assuming you have already set up the necessary Plotly JavaScript integration
						// Insert the Plotly embed code here
						echo '<div id="plotly-chart"></div>';
						echo '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>';
						echo '<script type="text/javascript">
							fetch("' . esc_url($plotly_json) . '")
								.then(response => response.json())
								.then(data => {
									Plotly.newPlot("plotly-chart", data.data, data.layout);
								});
						</script>';
					}

				endwhile;
			}

			/**
			 * generate_after_main_content hook.
			 *
			 * @since 0.1
			 */
			do_action( 'generate_after_main_content' );
			?>
		</main>
	</div>

	<?php
	/**
	 * generate_after_primary_content_area hook.
	 *
	 * @since 2.0
	 */
	do_action( 'generate_after_primary_content_area' );

	generate_construct_sidebars();

	get_footer();

	echo '<script type="text/javascript">
	    fetch("' . esc_url($plotly_json) . '")
		.then(response => {
		    console.log("Plotly JSON path:", "' . esc_url($plotly_json) . '");
		    return response.json();
		})
		.then(data => {
		    console.log("Plotly JSON data:", data);
		    Plotly.newPlot("plotly-chart", data.data, data.layout);
		})
		.catch(error => console.error("Error loading Plotly JSON:", error));
	</script>';
