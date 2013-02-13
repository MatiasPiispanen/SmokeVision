/**
 * @author Matias Piispanen
 */

var timerConsole;
var selected = 0;
var matrixStack = [];
var canvas;
var canvasContext;
var viewer;
var dim = 16;
var numCells;
var shaderProgram;
var shaderProgram2D;
var gl;

var platforms = [];
var devices = [];
var cl;
var clSrc;
var clQueue;
var devices;
var platforms;
var selectedPlatform;
var scalarProgram;
var vectorProgram;
var wgSize;
var localThreads;
var globalThreads;

var scalarAddKernel;
var scalarCopyKernel;
var scalarDiffusionKernel;
var scalarAdvectionKernel;
var scalarBoundariesKernel;
var volumeRayMarchingKernel;

var vectorAddKernel;
var vectorCopyKernel;
var vectorDiffusionKernel;
var vectorAdvectionKernel;
var vectorInitFieldKernel;
var vectorBoundariesKernel;
var vectorProjectionFirst;
var vectorProjectionSecond;
var vectorProjectionThird;
var vectorVorticityFirstKernel;
var vectorVorticitySecondKernel;

var scalarField;
var vectorField;
var scalarAddField;

var scalarBuffer;
var scalarSourceBuffer;
var scalarTempBuffer;
var scalarSecondTempBuffer;

var vectorBuffer;
var vectorSourceBuffer;
var vectorTempBuffer;

var pixelBuffer;
var pixelCount;

var scalarSrc;
var vectorSrc;

var mouseX = 0;
var mouseY = 0;
var mouseButton = 0;
var mousePressed = 0;

var dirX;
var dirY;

var clTime = 0;
var clMemTime = 0;
var jsTime = 0;
var raymarchTime = 0;
var prevTime = 0;

var viscosity = 0.00001;
var dt = 0.033;
var ds = 1.0;

var running = true;

requestAnimationFrame = window.requestAnimationFrame || window.mozRequestAnimationFrame ||  
  window.webkitRequestAnimationFrame || window.msRequestAnimationFrame;  

var start = window.mozAnimationStartTime;  // Only supported in FF. Other browsers can use something like Date.now(). 

function webclfluid() {
	var box;
	var boundaries = [];
	
	numCells = (dim+2)*(dim+2)*(dim+2);
	
	timerConsole = document.getElementById("test");
	canvas = document.getElementById("sim_canvas");
	
	document.getElementById("dt").value = dt;
	document.getElementById("ds").value = ds;
	document.getElementById("viscosity").value = viscosity;
	
	// Get WebGL context
	gl = initWebGL("sim_canvas");
	
	pixelCount = canvas.width*canvas.height;
	
	gl.viewport(0, 0, canvas.width, canvas.height);
	
	if(!gl) {
		return;
	}
  
  shaderProgram2D = simpleSetup( gl, "2d-vertex-shader", "2d-fragment-shader", [ "a_position", "a_texCoord"], [ 0, 0, 0, 0 ], 10000);
	
	// create scalar and vector fields
	scalarField = new ScalarField(dim, viscosity, dt, boundaries, box);
	vectorField = new VectorField(dim, viscosity, dt, boundaries);
  
	viewer = new Viewer(gl, shaderProgram, scalarField);
	
	// Set up WebCL
	if(window.WebCL == undefined) {
		alert("Your browser doesn't support WebCL");
		return false;
	}
	
	scalarSrc = getKernel("scalar_kernels.cl");
	vectorSrc = getKernel("vector_kernels.cl");
	
	// Init empty scalar and vector fields used as source fields
	scalarAddField = new Float32Array(numCells);
	vectorAddField = new Float32Array(numCells*3);
	
	for(var i = 0; i < dim+2; i++) {
		for(var j = 0; j < dim+2; j++) {
			for(var k = 0; k < dim+2; k++) {
				scalarAddField[index(i,j,k,dim)] = 0.0;
				vectorAddField[vindex(i,j,k,0,dim)] = 0.0;
				vectorAddField[vindex(i,j,k,1,dim)] = 0.0;
				vectorAddField[vindex(i,j,k,2,dim)] = 0.0;
			}
		}
	}
	
	// Density and Velocity added to the simulation on each cycle
	scalarAddField[index(dim/2,dim-1,dim/2,dim)] = 1000.0;
	vectorAddField[vindex(dim/2,dim-2,dim/2,1,dim)] = -200.0;
	
	setupWebCL();
	
	canvas.addEventListener('mousemove', mouse_move, false);
	canvas.addEventListener('mouseenter', mouse_enter, false);
	canvas.addEventListener('mouseleave', mouse_leave, false);
  mousePressed = 0;
	
	running = true;
	prevTime = Date.now();
	requestAnimationFrame(step, canvas); 
}

function mouse_move(e) {
	dirX = (e.layerX - mouseX) * 100;
	mouseX = e.layerX;
	
	dirY = (e.layerY - mouseY) * 100;
	mouseY = e.layerY;

	if(mousePressed === 1) {
		var simX = Math.floor(e.layerX * (dim / canvas.clientWidth));
		var simY = Math.floor(e.layerY * (dim / canvas.clientHeight));
		
		if(dirX > 1000.0) {
			dirX = 1000.0;
		}
		else if(dirX < -1000.0) {
			dirX = -1000.0;
		}
		
		if(dirY > 1000.0) {
			dirY = 1000.0;
		}
		else if(dirY < -1000.0) {
			dirY = -1000.0;
		}
		for(var i = 1; i < dim+1; i++) {
			vectorAddField[vindex(simX,simY,i,0,dim)] = dirX;
			vectorAddField[vindex(simX,simY,i,1,dim)] = dirY;
		}
		
		var start = Date.now();
		var bufSize = 4 * numCells;
		clQueue.enqueueWriteBuffer(vectorSourceBuffer, true, 0, bufSize*3, vectorAddField, []);
		clMemTime = Date.now() - start;
		
		for(var i = 1; i < dim+1; i++) {
			vectorAddField[vindex(simX,simY,i,0,dim)] = 0.0;
			vectorAddField[vindex(simX,simY,i,1,dim)] = 0.0;
		}
	}
	else {
		var start = Date.now();
		var bufSize = 4 * numCells;
		clQueue.enqueueWriteBuffer(vectorSourceBuffer, true, 0, bufSize*3, vectorAddField, []);
		clMemTime = Date.now() - start;
	}
}

function mouse_enter(e) {
  mousePressed = 1;
}

function mouse_leave(e) {
  mousePressed = 0;
}

function step() {  
	if(running == true) {		
		jsTime = Date.now() - prevTime - clTime - clMemTime - raymarchTime;
		prevTime = Date.now();
		timerConsole.innerHTML  = "<br>WebCL  (ms): " + clTime;
    timerConsole.innerHTML += "<br>WebCL memory transfers (ms): " + clMemTime;
    timerConsole.innerHTML += "<br>Raymarch (ms): " + raymarchTime;
    timerConsole.innerHTML += "<br>JavaScript (ms): " + jsTime;
		
		jsTime = 0;
		clTime = 0;
		raymarchTime = 0;
		clMemTime = 0;
		scalarField.step(scalarAddField);
		vectorField.step(vectorAddField);
		viewer.draw();
		
		requestAnimationFrame(step, canvas);  
	}
}  

function clDeviceQuery() {
  var deviceList = [];
  var platforms = (window.WebCL && WebCL.getPlatforms()) || [];
  for (var p=0, i=0; p < platforms.length; p++) {
    var plat = platforms[p];
    var devices = plat.getDevices(WebCL.CL_DEVICE_TYPE_ALL);
    for (var d=0; d < devices.length; d++, i++) {
      deviceList[i] = { 'device' : devices[d], 
                        'type' : devices[d].getDeviceInfo(WebCL.CL_DEVICE_TYPE),
                        'name' : devices[d].getDeviceInfo(WebCL.CL_DEVICE_NAME),
                        'version' : devices[d].getDeviceInfo(WebCL.CL_DEVICE_VERSION),
                        'vendor' : plat.getPlatformInfo(WebCL.CL_PLATFORM_VENDOR),
                        'platform' : plat };
    }
  }
  console.log(deviceList);
  return deviceList;
}

function setupWebCL() {

  var deviceList = clDeviceQuery();

  if (deviceList.length === 0) {
		alert("Unfortunately your browser/system doesn't support WebCL.");
		return false;
	}

  try {
    var htmlDeviceList = "";
		for(var i in deviceList) {
			htmlDeviceList += "<option value=" + i + ">" + deviceList[i].vendor + ": " + deviceList[i].name + "</option>\n";
		}

		var deviceselect = document.getElementById("devices");
		deviceselect.innerHTML = htmlDeviceList;
		deviceselect.selectedIndex = selected;

    var selectedDevice = deviceList[selected].device;
    var selectedPlatform = deviceList[selected].platform;
    cl = WebCL.createContext([WebCL.CL_CONTEXT_PLATFORM, selectedPlatform], [selectedDevice]);
		clQueue = cl.createCommandQueue(selectedDevice, null);
		allocateBuffers();
	} catch(err) {
		alert("Error initializing WebCL");
    return false;
	}

	try {
		scalarProgram = cl.createProgramWithSource(scalarSrc);
    var program = scalarProgram;
		program.buildProgram([selectedDevice], "-cl-fast-relaxed-math -cl-denorms-are-zero");

		vectorProgram = cl.createProgramWithSource(vectorSrc);
    program = vectorProgram;
		program.buildProgram([selectedDevice], "-cl-fast-relaxed-math -cl-denorms-are-zero");
	} catch(e) {
		console.log("Failed to build WebCL program. Error " + 
          program.getProgramBuildInfo(selectedDevice, WebCL.CL_PROGRAM_BUILD_STATUS) + ":  " +
          program.getProgramBuildInfo(selectedDevice, WebCL.CL_PROGRAM_BUILD_LOG));
	}
	
	scalarAddKernel = scalarProgram.createKernel("scalarAddField");
	scalarCopyKernel = scalarProgram.createKernel("scalarCopy");
	scalarDiffusionKernel = scalarProgram.createKernel("scalarDiffusion");
	scalarAdvectionKernel = scalarProgram.createKernel("scalarAdvection");
	scalarBoundariesKernel = scalarProgram.createKernel("scalarBoundaryDensities");
	volumeRayMarchingKernel = scalarProgram.createKernel("volumeRayMarching");
	
	vectorAddKernel = vectorProgram.createKernel("vectorAddField");
	vectorCopyKernel = vectorProgram.createKernel("vectorCopy");
	vectorAdvectionKernel = vectorProgram.createKernel("vectorAdvection");
	vectorDiffusionKernel = vectorProgram.createKernel("vectorDiffusion");
	vectorInitFieldKernel = vectorProgram.createKernel("vectorInitField");
	vectorBoundariesKernel = vectorProgram.createKernel("vectorBoundaries");
	vectorProjectionFirst = vectorProgram.createKernel("vectorProjectionFirst");
	vectorProjectionSecond = vectorProgram.createKernel("vectorProjectionSecond");
	vectorProjectionThird = vectorProgram.createKernel("vectorProjectionThird");
	vectorVorticityFirstKernel = vectorProgram.createKernel("vectorVorticityConfinementFirst");
	vectorVorticitySecondKernel = vectorProgram.createKernel("vectorVorticityConfinementSecond");
	
	localThreads = (Math.ceil((dim + 2) / 32)) * 32;
}

function reset() {
	scalarField.reset();
	vectorField.reset();
}

function stop() {
	if(running) {
		running = false;
		document.getElementById("stop").innerHTML = "Start";
	}
	else {
		running = true;
		requestAnimationFrame(step, canvas);  
		document.getElementById("stop").innerHTML = "Stop";
	}
}

function dtChanged(value) {
	if(!isNaN(value) && value > 0) {
		dt = value;
		scalarField.setTimestep(dt);
		vectorField.setTimestep(dt);
	}
	else {
		document.getElementById("dt").value = dt;
	}
}

function dsChanged(value) {
	if(!isNaN(value) && value > 0) {
		ds = value;
	}
	else {
		document.getElementById("ds").value = ds;
	}
}

function viscosityChanged(value) {
	if(!isNaN(value) && value > 0) {
		viscosity = value;
		scalarField.setViscosity(viscosity);
		vectorField.setViscosity(viscosity);
	}
	else {
		document.getElementById("viscosity").value = viscosity;
	}
}

function allocateBuffers() {
	var bufSize = 4 * numCells;
	
	pixelBuffer = cl.createBuffer(WebCL.CL_MEM_READ_ONLY, pixelCount * 4);
	
	/* Scalar Buffers */
	scalarBuffer = cl.createBuffer(WebCL.CL_MEM_READ_WRITE, bufSize);
	clQueue.enqueueWriteBuffer(scalarBuffer, true, 0, bufSize, scalarField.getField(), []);
	
	scalarSourceBuffer = cl.createBuffer(WebCL.CL_MEM_READ_ONLY, bufSize);
	clQueue.enqueueWriteBuffer(scalarSourceBuffer, true, 0, bufSize, scalarAddField, []);
	
	scalarTempBuffer= cl.createBuffer(WebCL.CL_MEM_READ_WRITE, bufSize);
	scalarSecondTempBuffer= cl.createBuffer(WebCL.CL_MEM_READ_WRITE, bufSize);
	
	/* Vector Buffers */
	vectorBuffer = cl.createBuffer(WebCL.CL_MEM_READ_WRITE, bufSize*3);
	clQueue.enqueueWriteBuffer(vectorBuffer, true, 0, bufSize*3, vectorField.getField(), []);
	
	vectorSourceBuffer = cl.createBuffer(WebCL.CL_MEM_READ_ONLY, bufSize*3);
	clQueue.enqueueWriteBuffer(vectorSourceBuffer, true, 0, bufSize*3, vectorAddField, []);
	
	vectorTempBuffer= cl.createBuffer(WebCL.CL_MEM_READ_WRITE, bufSize*3);
}

function freeBuffers() {
	pixelBuffer.releaseCLResources();
	scalarBuffer.releaseCLResources();
	scalarSourceBuffer.releaseCLResources();
	scalarTempBuffer.releaseCLResources();
	scalarSecondTempBuffer.releaseCLResources();
	vectorBuffer.releaseCLResources();
	vectorSourceBuffer.releaseCLResources();
	vectorTempBuffer.releaseCLResources();
	
	cl.releaseCLResources();
	clQueue.releaseCLResources();
	scalarProgram.releaseCLResources();
	vectorProgram.releaseCLResources();
	
	vectorAddKernel.releaseCLResources();;
	vectorCopyKernel.releaseCLResources();;
	vectorDiffusionKernel.releaseCLResources();;
	vectorAdvectionKernel.releaseCLResources();;
	vectorInitFieldKernel.releaseCLResources();;
	vectorBoundariesKernel.releaseCLResources();;
	vectorProjectionFirst.releaseCLResources();;
	vectorProjectionSecond.releaseCLResources();;
	vectorProjectionThird.releaseCLResources();;
	vectorVorticityFirstKernel.releaseCLResources();;
	vectorVorticitySecondKernel.releaseCLResources();;
}

function simResolutionChanged(resolution) {
	running = false;
	
	if(resolution == 0) {
		dim = 16
	}
	else if(resolution == 1) {
		dim = 32;
	}
	else if(resolution == 2) {
		dim = 64;
	}
	else if(resolution == 3) {
		dim = 96;
	}
	else if(resolution == 4) {
		dim = 128;
	}
	
	freeBuffers();
	webclfluid();
}

function resolutionChanged(resolution) {
	running = false;
	
	if(resolution == 0) {
		canvas.width = 320;
		canvas.height = 240;
	}
	else if(resolution == 1) {
		canvas.width = 640;
		canvas.height = 480;
	}
	else if(resolution == 2) {
		canvas.width = 800;
		canvas.height = 600;
	}
	else if(resolution == 3) {
		canvas.width = 1024;
		canvas.height = 768;
	}
	
	freeBuffers();
	webclfluid();
}

function deviceChanged(device) {
	running = false;
	freeBuffers();
	
	selected = device;
	
	webclfluid();
}

function pushMatrix(viewer) {
	var copy = mat4.create();
	mat4.set(viewer.mvMatrix,copy);
	matrixStack.push(copy);
}

function popMatrix(viewer) {
	if(matrixStack.length > 0) {
		viewer.mvMatrix = matrixStack.pop();
	}
}

function getKernel(src) {
  var xhr = new XMLHttpRequest();
  xhr.open("GET", src, false);
  xhr.send(null);
  if (xhr.status == 200) {
    return xhr.responseText;
  } else {
    console.log("XMLHttpRequest error!");
    return null;
  }
}
