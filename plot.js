var data1 = new	TimeSeries();
var data2 = new	TimeSeries();
var position = new TimeSeries();
var velocity = new TimeSeries();


function createPlot2() {
	var chart = new SmoothieChart({
		grid: { strokeStyle:'rgb(2, 2, 2)', fillStyle:'rgb(255, 255, 255)',
        lineWidth: 0.1, millisPerLine: 1000, verticalSections: 5, },
        //maxValue:10,minValue:-1
	});
	chart.addTimeSeries(data1, { strokeStyle:'rgb(55, 0, 125)', lineWidth:1 });
	chart.addTimeSeries(data2, { strokeStyle:'rgb(255,45,151)', lineWidth:1 });
	chart.streamTo(document.getElementById("chart2"), 200);
}

function createPlot() {
	var chart = new SmoothieChart({
		grid: { strokeStyle:'rgb(2, 2, 2)', fillStyle:'rgb(255, 255, 255)',
        lineWidth: 0.1, millisPerLine: 1000, verticalSections: 5, },
        //maxValue:10,minValue:-1
	});
	chart.addTimeSeries(position, { strokeStyle:'rgb(55, 0, 125)', lineWidth:1 });
	chart.addTimeSeries(velocity, { strokeStyle:'rgb(255,45,151)', lineWidth:1 });
	chart.streamTo(document.getElementById("chart"), 200);
}

function createPlots(){
	createPlot();
	createPlot2();
}

