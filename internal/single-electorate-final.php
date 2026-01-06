<?php get_header(); ?>

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
    echo '';
  }

  // Get the bar plot JSON file
  $bar_file = get_field('FP_bar_plots');
  if ( ! $bar_file ) {
    error_log('❌ No bar plot data available for ' . get_the_title());
    echo '';
  }
?>

<div style="width:100vw; padding:2rem 0; background:#f2e7fe;">
  <h1 style="font-family: 'Georgia', serif; font-size: 3rem; text-align: center; margin: 0 0 2rem; color: #333;">
    <?php the_title(); ?>
  </h1>

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

  <div id="bar-plot" style="width: 90%; max-width: 1400px; margin: 0 auto 3rem auto;"></div>

  <div id="violin-plot" style="width: 90%; max-width: 1400px; margin: 0 auto;"></div>
  
  <div id="table-container" style="margin-top: 2rem; width: 100%; display: flex; justify-content: center; padding-bottom: 20px;">
    <div style="max-width: 1200px; width: 100%; padding: 0 20px;">
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
     const layout = { 
       ...(originalLayout || {}), 
       autosize: true, 
       paper_bgcolor: 'rgba(0,0,0,0)', 
       plot_bgcolor: 'rgba(0,0,0,0)', 
       xaxis: { 
         visible: false, 
         fixedrange: true  // Prevents zooming on x-axis 
       }, 
       yaxis: { 
         visible: false, 
         fixedrange: true  // Prevents zooming on y-axis 
       }, 
       dragmode: false,  // Disables pan/zoom interaction 
       title: { 
         text: 'Probabilities of Winning', 
         font: { 
           size: screenWidth < 600 ? 16 : screenWidth < 1024 ? 22 : 26, 
           color: 'black', 
         }, 
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
       displayModeBar: 'hover' 
     }); 
   }) 
   .catch(err => { 
     console.error('Bar plot load error:', err); 
     document.getElementById('bar-plot').innerHTML = '<p style="text-align:center; color:red; margin-top:2rem;">❌ Failed to load bar plot.</p>'; 
   }); 

 // Violin Plot
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
       const layout = { 
         ...(originalLayout || {}), 
         autosize: true, 
         height: window.innerHeight * 0.75, 
         responsive: true, 
         paper_bgcolor: 'rgba(0,0,0,0)', 
         plot_bgcolor: 'rgba(0,0,0,0)', 
         title: { 
             ...(originalLayout?.title || {}), 
             text: "Distribution of First Preference Votes", 
             font: { 
                size: screenWidth < 600 ? 18 : screenWidth < 1024 ? 24 : 30, 
                family: 'Arial, sans-serif', 
                color: 'rgb(0,0,0)' 
             }, 
             y: 0.99, 
             yanchor: 'top', 
         }, 
         xaxis: { 
           ...(originalLayout?.xaxis || {}), 
           automargin: true, 
           showticklabels: screenWidth > 450, 
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
         margin: { t: 40, b: 40, l: 40, r: 40 }, 
         modeBarButtonsToRemove: ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'], 
         dragmode: false, 
         mobile: { 
            dragmode: false, 
         }, 
       }; 
        
       // Ensure tooltips show y-value + trace name 
       data.forEach(trace => { 
          console.log("Trace name:", trace.name); 
          trace.hoverinfo = 'name+y'; 
          trace.hoveron = 'points'; 
          // Clean name to remove " sims" 
          const partyName = trace.name.replace(/\s*sims\s*$/i, ''); 
          // Set a custom hover template 
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
            displayModeBar: 'hover' 
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
                color: Array.isArray(trace.marker?.color) ? trace.marker.color : Array(len).fill(trace.marker?.color || 'blue'), 
                opacity: Array.isArray(trace.marker?.opacity) ? trace.marker.opacity : Array(len).fill(trace.marker?.opacity ?? 0), 
                size: Array.isArray(trace.marker?.size) ? trace.marker.size : Array(len).fill(trace.marker?.size ?? 6) 
              }; 
            }); 
            initialized = true; 
            // ✅ Handle browser resizing 
            window.addEventListener('resize', () => { 
              if (initialized) { 
                Plotly.Plots.resize(violinPlotElement); 
                Plotly.Plots.resize(document.getElementById('bar-plot')); 
              } 
            }); 
          }).catch(err => { 
            console.error('Plot load error:', err); 
            document.getElementById('violin-plot').innerHTML = '<p style="text-align:center; color:red; margin-top:2rem;">❌ Failed to load violin plot.</p>'; 
          }); 
       } else { 
          Plotly.relayout('violin-plot', layout); 
       } 
     }; 
     updateLayout(); // <--- ✅ call it immediately to render first time 
   }) 
   .catch(err => { 
     console.error('Violin plot load error:', err); 
     document.getElementById('violin-plot').innerHTML = '<p style="text-align:center; color:red; margin-top:2rem;">❌ Failed to load violin plot.</p>'; 
   }); 
</script> 
<?php get_footer(); ?>