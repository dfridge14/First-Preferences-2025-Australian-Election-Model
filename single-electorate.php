<?php get_header(); ?>

<!-- Force update -->


<?php
// Fetch all electorate posts
$electorates = get_posts([
    'post_type' => 'electorate',
    'numberposts' => -1,
    'orderby' => 'title',
    'order' => 'ASC',
]);
?>


<?php 
  // Get the violin plot JSON file
  $violin_file = get_field('FP_plots');
  if ( ! $violin_file ) {
    error_log('❌ No violin plot data available for ' . get_the_title());
    echo '<!-- ❌ No violin plot data available -->';
    // Optionally skip plotting JS or add fallback here
  }

  // Get the bar plot JSON file
  $bar_file = get_field('FP_bar_plots');
  if ( ! $bar_file ) {
    error_log('❌ No bar plot data available for ' . get_the_title());
    echo '<!-- ❌ No bar plot data available -->';
    // Optionally skip plotting JS or add fallback here
  }
?>

<div class="electorate-wrap">
<h1 class="electorate-title"><?php the_title(); ?></h1>

  <!-- ✅ Insert dropdown here -->
  <div style="text-align:right; margin: 0 5% 2rem 5%;">
    <select id="electorate-select" onchange="if (this.value) window.location.href=this.value;" style="padding: 0.5rem; font-size: 1rem; min-width: 200px;">
      <option value="">-- Choose Electorate --</option>
      <?php foreach ($electorates as $electorate): ?>
        <option value="<?php echo get_permalink($electorate->ID); ?>" <?php selected(get_the_ID(), $electorate->ID); ?>>
          <?php echo esc_html($electorate->post_title); ?>
        </option>
      <?php endforeach; ?>
    </select>
  </div>

  <!-- Div for Bar Plot -->
  <div id="bar-plot" style="
      width: 100%;
      max-width: 1400px;
      margin: 0 auto 3rem auto;
  "></div>

  <!-- Div for Violin Plot -->
  <div id="violin-plot" style="
      width: 100%;
      max-width: 1400px;
      margin: 0 auto;
  "></div>
  
<!-- Insert the table from post content -->
<div id="table-container" style="margin-top: 2rem; width: 100%; display: flex; justify-content: center; padding-bottom: 20px;">
  <div style="max-width: 1200px; width: 100%; padding: 0">
    <p style="text-align:center; color:blue;"> </p>
    <?php the_content(); ?>
  </div>
</div>

</div>

<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>

 // Bar Plot
console.log('⭑ Loading bar plot JSON from:', '<?php echo esc_url($bar_file); ?>');
fetch('<?php echo esc_url($bar_file); ?>')
  .then(r => {
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  })
  .then(({ data, layout: originalLayout }) => {
    const screenWidth = window.innerWidth;

    const side = screenWidth < 768 ? 5 : 40; // padding on side of barplot

    const layout = {
      ...(originalLayout || {}),
      autosize: true,
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      xaxis: {
        visible: false,
        fixedrange: true,  // Prevents zooming on x-axis
        range: [0, 1],          // or your known max
        constrain: 'domain'
      },
      yaxis: {
        visible: false,
        fixedrange: true  // Prevents zooming on y-axis
      },
      dragmode: false,  // Disables pan/zoom interaction
      
      title: {
        text: 'Probabilities of Winning',
        x: 0,
        xanchor: 'left',
        xref: 'paper',
        pad: { l: 0 },
        font: {
          family: 'Arial, sans-serif',
          size: screenWidth < 600 ? 18 : screenWidth < 1024 ? 24 : 30,
          color: 'black'
        },
      },
      margin: {
        t: 40,
        b: 40,
        l: side,
        r: side
      },
    };

    Plotly.newPlot('bar-plot', data, layout, {
      modeBarButtonsToRemove: [
        'zoom2d', 'pan2d', 'select2d', 'lasso2d',
        'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'
      ],
      displaylogo: false,
      scrollZoom: false,  // Disables zooming with scroll
      doubleClick: false, // Prevents double-click zoom/reset
      responsive: true,
      displayModeBar: (screenWidth >= 768) ? 'hover' : false
    });
  })
  .catch(err => {
    console.error('Bar plot load error:', err);
    document.getElementById('bar-plot').innerHTML = '<p style="text-align:center; color:red; margin-top:2rem;">❌ Failed to load bar plot.</p>';
  });


  console.log('⭑ Loading plot JSON from:', '<?php echo esc_url($violin_file); ?>');

fetch('<?php echo esc_url($violin_file); ?>')
  .then(r => {
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  })
  .then(({ data, layout: originalLayout }) => {
    const violinPlotElement = document.getElementById('violin-plot');
    let initialized = false;
    const hoverIndexMap = {};  // simID => array of matching points
    let origMarker = [];
    let lastSimID = null;
    let hoverTimeout = null;
    
    let resizeTimeout = null;



    const updateLayout = () => {
      const screenWidth = window.innerWidth;

      const isMid = screenWidth < 600; // for medium screen with too many candidates
      const nCats = new Set(
        data.flatMap(t => Array.isArray(t.x) ? t.x : [])
      ).size;
      const manyCats = nCats >= 10;

      const left = (screenWidth < 768) ? 10 : 30;
      const right = (screenWidth < 768) ? 5 : 10; // for small/med screen, remove padding

      const layout = {
        ...(originalLayout || {}),
        autosize: true,
        height: window.innerHeight * 0.75,
        responsive: true,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
         title: {
	      ...(originalLayout?.title || {}),
	      text: "Distribution of First Preference Votes", // Add your title text here
	      font: {
          family: 'Arial, sans-serif',
          size: screenWidth < 600 ? 18 : screenWidth < 1024 ? 24 : 30,
          color: 'black'
        },
	      y: 0.99, // Adjust the vertical position to give space above the plot
	      yanchor: 'top', // Anchors the title at the top
	    },
        xaxis: {
          ...(originalLayout?.xaxis || {}),
          automargin: true,
          showticklabels: !(isMid && manyCats), // will need to adjust for many candidate electorates
          title: screenWidth > 1024 ? (originalLayout?.xaxis?.title || '') : '',
        },
        yaxis: {
          ...(originalLayout?.yaxis || {}),
          automargin: true,
        },
        legend: {
          ...(originalLayout?.legend || {}),
          orientation: 'v',
          x: 1,
          y: 1,
          xanchor: 'right',
          yanchor: 'top',
          font: {
            size: screenWidth > 1600 ? 18 : screenWidth > 1024 ? 14 : 10
          },
          itemclick: false,
          itemdoubleclick: false
        },
        margin: {
          t: 40,
          b: 40,
          l: left,
          r: right
        },
        modeBarButtonsToRemove: ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
        dragmode: false,
         mobile: {
	    dragmode: false,  // Disable touch dragmode (box-select and lasso-select)
	    //showglow: false,  // Disable the glow effect on touch devices (optional)
	    //displayModeBar: true, // Show the mode bar if needed
	  },
      };

    const isMobile = screenWidth < 768;

    // Ensure tooltips show y-value + trace name
    data.forEach(trace => {
        console.log("Trace name:", trace.name); 

        // 🔹 MOBILE ONLY: make violins fatter
        if (isMobile && trace.type === 'violin') {
          trace.width = 0.7;
        }

        // CASE A: Big hitbox trace (no tooltip ever)
        if (trace.name && trace.name.startsWith("Hitbox::")) {
          trace.hoverinfo = 'skip';
          return;
        }

        // CASE B: Small result dot (tooltip should show)
        if (trace.name && trace.name.startsWith("Result::")) {
          const width = window.innerWidth;
          let dotSize = 15;
          if (width < 600) dotSize = 10;
          else if (width < 1200) dotSize = 15;
          else dotSize = 20;

          if (!trace.marker) trace.marker = {};
          trace.marker.size = dotSize;

          // IMPORTANT: allow tooltip on the dot trace
          return;
        }

        // CASE C: Standard Traces
        trace.hoverinfo = 'name+y';  
        trace.hoveron = 'points';    
        const partyName = trace.name.replace(/\s*sims\s*$/i, '');
        trace.hovertemplate = `
          <b style="text-align:center">${partyName}</b><br>
          <b style="text-align:center">%{y:.1f}%</b><extra></extra>
        `;
      });
	console.log("Plot Data:", data);

      if (!initialized) {
	  Plotly.newPlot('violin-plot', data, layout, {
	    modeBarButtonsToRemove: [
	      'zoom2d', 'pan2d', 'select2d', 'lasso2d',
	      'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'
	    ],
	    displaylogo: false,
	    responsive: true,
	    displayModeBar: (screenWidth >= 768) ? 'hover' : false
	  }).then(() => {
	    const plotData = violinPlotElement.data;

	    // Build hover index map
	    plotData.forEach((trace, traceIndex) => {
	      if (!trace.customdata) return;
	      trace.customdata.forEach((simID, pointIndex) => {
		if (!hoverIndexMap[simID]) hoverIndexMap[simID] = [];
		hoverIndexMap[simID].push({ traceIndex, pointIndex });
	      });
	    });

	    // Store original marker values
	    origMarker = plotData.map(trace => {
	      const len = trace.y.length;
	      return {
		color: Array.isArray(trace.marker?.color)
		  ? trace.marker.color
		  : Array(len).fill(trace.marker?.color || 'blue'),
		opacity: Array.isArray(trace.marker?.opacity)
		  ? trace.marker.opacity
		  : Array(len).fill(trace.marker?.opacity ?? 0),
		size: Array.isArray(trace.marker?.size)
		  ? trace.marker.size
		  : Array(len).fill(trace.marker?.size ?? 6)
	      };
	    });

		initialized = true;

    // === Forward hover from big invisible hitbox -> small visible dot ===
    const gd = violinPlotElement;

    function findTraceIndexByName(name) {
      return gd.data.findIndex(t => t && t.name === name);
    }

    gd.on('plotly_hover', function (evt) {
      const pt = evt.points && evt.points[0];
      if (!pt || !pt.data || !pt.data.name) return;

      const traceName = pt.data.name;

      // Only react when hovering the big hitbox trace
      if (!traceName.startsWith("Hitbox::")) return;

      const party = traceName.slice("Hitbox::".length);
      const dotTraceName = `Result::${party}`;
      const dotTraceIndex = findTraceIndexByName(dotTraceName);

      if (dotTraceIndex === -1) return;

      Plotly.Fx.hover(gd, [{ curveNumber: dotTraceIndex, pointNumber: 0 }], "closest");
    });

    gd.on('plotly_unhover', function () {
      Plotly.Fx.unhover(gd);
    });


	    // ✅ Handle browser resizing
	    window.addEventListener('resize', () => {
	      if (initialized) {
		Plotly.Plots.resize(violinPlotElement);
		Plotly.Plots.resize(document.getElementById('bar-plot'));
	      }
	    });
	  }).catch(err => {
	    console.error('Plot load error:', err);
	    document.getElementById('violin-plot').innerHTML =
	      '<p style="text-align:center; color:red; margin-top:2rem;">❌ Failed to load violin plot.</p>';
	  });
	} else {
	  Plotly.relayout('violin-plot', layout); // Relayout if already initialized
	}
    };
    updateLayout(); // <--- ✅ call it immediately to render first time
  })
  .catch(err => {
    console.error('Violin plot load error:', err);
    document.getElementById('violin-plot').innerHTML =
      '<p style="text-align:center; color:red; margin-top:2rem;">❌ Failed to load violin plot.</p>';
  });


</script>

<?php get_footer(); ?>
