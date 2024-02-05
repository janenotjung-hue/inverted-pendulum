function init() {
  initHost('mycanvas');
}

var seriesOptions = [
  { strokeStyle: 'rgba(255, 0, 0, 1)', fillStyle: 'rgba(255, 0, 0, 0.1)', lineWidth: 3 },
  { strokeStyle: 'rgba(0, 255, 0, 1)', fillStyle: 'rgba(0, 255, 0, 0.1)', lineWidth: 3 },
  { strokeStyle: 'rgba(0, 0, 255, 1)', fillStyle: 'rgba(0, 0, 255, 0.1)', lineWidth: 3 },
  { strokeStyle: 'rgba(255, 255, 0, 1)', fillStyle: 'rgba(255, 255, 0, 0.1)', lineWidth: 3 }
];

function initHost(hostId) {

  // Initialize an empty TimeSeries for each CPU.
  var cpuDataSets = [new TimeSeries(), new TimeSeries()];

  // Every second, simulate a new set of readings being taken from each CPU.
  setInterval(function() {
    getTheta(Date.now(), cpuDataSets);
  }, 1);

  // Build the timeline
  var timeline = new SmoothieChart({ fps: 30, millisPerPixel: 20, grid: { strokeStyle: '#555555', lineWidth: 1, millisPerLine: 1000, verticalSections: 4}, tooltip: true});
  for (var i = 0; i < cpuDataSets.length; i++) {
    timeline.addTimeSeries(cpuDataSets[i], seriesOptions[i]);
  }
  timeline.streamTo(document.getElementById(hostId), 1);
}

function getTheta(time, dataSets) {
  for (var i = 0; i < dataSets.length; i++) {
    dataSets[i].append(time, document.getElementById('thetaText').innerHTML);
  }
}
