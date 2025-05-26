<script lang="ts">
  import { onMount } from 'svelte';
  import { Chart, BarController, BarElement, CategoryScale, LinearScale, Tooltip, Legend } from 'chart.js';

  Chart.register(BarController, BarElement, CategoryScale, LinearScale, Tooltip, Legend);

  export let data: Record<string, number> = {};
  let canvas: HTMLCanvasElement;

  let chart: Chart;

  onMount(() => {
    const labels = Object.keys(data);
    const values = Object.values(data);
    const barColors = labels.map((_, i) => `hsl(${(i * 360) / labels.length}, 70%, 60%)`);

    chart = new Chart(canvas, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: '# of Votes',
          data: values,
          backgroundColor: barColors
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
        },
        scales: {
          y: {
            beginAtZero: true,
            ticks: {
              precision: 0
            }
          }
        }
      }
    });

    return () => chart.destroy();
  });
</script>

<canvas bind:this={canvas}></canvas>

<style>
  canvas {
    max-width: 500px;
    margin: 2rem auto;
    display: block;
  }
</style>
