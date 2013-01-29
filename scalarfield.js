var scale = [];
var corner = [];
var pixels;

function ScalarField(dim, viscosity, dt, boundaries, box) {
	this.field = new Float32Array(getFlatSize(dim));
	this.box = box;
	
	this.dim = dim;
	this.viscosity = viscosity;
	this.dt = dt;
	this.texture = gl.createTexture();
	
	scale = [1/(this.dim+2)*2, 1/(this.dim+2)*2, 1/(this.dim+2)*2];
	corner = [-(this.dim+2)/2, -(this.dim+2)/2, -(this.dim+2)/2];
	
	for(var i = 0; i < this.dim+2; i++) {
		for(var j = 0; j < this.dim+2; j++) {
			for(var k = 0; k < this.dim+2; k++) {
				this.field[index(i,j,k,dim)] = 0.0;
			}
		}
	}
}

ScalarField.prototype.setTimestep = function(value) {
	this.dt = value;
}

ScalarField.prototype.setViscosity = function(value) {
	this.viscosity = value;
}

ScalarField.prototype.reset = function() {
	var bufSize = 4 * numCells;
	
	for(var i = 0; i < this.dim+2; i++) {
		for(var j = 0; j < this.dim+2; j++) {
			for(var k = 0; k < this.dim+2; k++) {
				this.field[index(i,j,k,dim)] = 0.0;
			}
		}
	}
	clQueue.enqueueWriteBuffer(scalarBuffer, true, 0, bufSize, this.getField(), []);
}

ScalarField.prototype.draw = function(viewer) {
	
	volumeRayMarchingKernel.setKernelArg(0, pixelBuffer);
	volumeRayMarchingKernel.setKernelArg(1, scalarBuffer);
	volumeRayMarchingKernel.setKernelArg(2, canvas.width, WebCL.types.UINT);
	volumeRayMarchingKernel.setKernelArg(3, canvas.height, WebCL.types.UINT);
	volumeRayMarchingKernel.setKernelArg(4, (canvas.height/2) / Math.tan(Math.PI/8), WebCL.types.FLOAT);
	volumeRayMarchingKernel.setKernelArg(5, -cubePos, WebCL.types.FLOAT);
	volumeRayMarchingKernel.setKernelArg(6, 2.0, WebCL.types.FLOAT); // TODO: magic number.
	volumeRayMarchingKernel.setKernelArg(7, this.dim, WebCL.types.UINT);
	volumeRayMarchingKernel.setKernelArg(8, ds, WebCL.types.FLOAT);
	
	try {
		var localWS = [];
		var globalWS = [Math.ceil(canvas.width / 32) * 32, Math.ceil(canvas.height / 32) * 32];
		pixels = new Uint8Array(pixelCount * 4);
		var start = Date.now();
		clQueue.enqueueNDRangeKernel(volumeRayMarchingKernel, globalWS.length, [], globalWS, localWS, []);
		clQueue.enqueueReadBuffer(pixelBuffer, true, 0, pixelCount * 4, pixels, []);
		clQueue.finish();
		raymarchTime += Date.now() - start;
		gl.useProgram(shaderProgram2D);
	}
	catch(e) {
		console.innerHTML = e;
	}
	
	shaderProgram2D.positionLocation = gl.getAttribLocation(shaderProgram2D, "a_position");
	shaderProgram2D.texCoordLocation = gl.getAttribLocation(shaderProgram2D, "a_texCoord");
  
  var texCoordBuffer = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
	gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
	  0.0,  0.0,
	  1.0,  0.0,
	  0.0,  1.0,
	  0.0,  1.0,
	  1.0,  0.0,
	  1.0,  1.0]), gl.STATIC_DRAW);
	gl.enableVertexAttribArray(shaderProgram2D.texCoordLocation);
	gl.vertexAttribPointer(shaderProgram2D.texCoordLocation, 2, gl.FLOAT, false, 0, 0);
	
	
	var texture = gl.createTexture();
	gl.bindTexture(gl.TEXTURE_2D, texture);
	
	// Set the parameters so we can render any size image.
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
	for(var i = 0; i < pixelCount; i++) {
		pixels[i*4 + 3] = 255;
	}
	
	// Upload the image into the texture.
	gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, canvas.width, canvas.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
	
	// lookup uniforms
	var resolutionLocation = gl.getUniformLocation(shaderProgram2D, "u_resolution");
	
	// set the resolution
	gl.uniform2f(resolutionLocation, canvas.width, canvas.height);
	
	var buffer = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
	gl.enableVertexAttribArray(shaderProgram2D.positionLocation);
	gl.vertexAttribPointer(shaderProgram2D.positionLocation, 2, gl.FLOAT, false, 0, 0);

	// Set a rectangle the same size as the image.
	setRectangle(gl, 0, 0, canvas.width, canvas.height);

	// Draw the rectangle.
	gl.drawArrays(gl.TRIANGLES, 0, 6);
}

function setRectangle(gl, x, y, width, height) {
  var x1 = x;
  var x2 = x + width;
  var y1 = y;
  var y2 = y + height;
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    x1, y1,
    x2, y1,
    x1, y2,
    x1, y2,
    x2, y1,
    x2, y2]), gl.STATIC_DRAW);
}

ScalarField.prototype.getField = function() {
	return this.field;
}

ScalarField.prototype.step = function(source) {
	var bufSize = 4 * numCells;
	this.addField(source);
	this.diffusion();
	this.advection();
	
	var start = Date.now();
	clQueue.enqueueReadBuffer(scalarBuffer, true, 0, bufSize, this.field, []);
	clMemTime = Date.now() - start;
}

ScalarField.prototype.addField = function(source) {
	scalarAddKernel.setKernelArg(0, scalarBuffer);
	scalarAddKernel.setKernelArg(1, scalarSourceBuffer);
	scalarAddKernel.setKernelArg(2, this.dim, WebCL.types.UINT);
	scalarAddKernel.setKernelArg(3, this.dt, WebCL.types.FLOAT);
	
	try {
		var localWS = [];
		var globalWS = [localThreads, localThreads, localThreads];
		
		var start = Date.now();
		clQueue.enqueueNDRangeKernel(scalarAddKernel, globalWS.length, [], globalWS, localWS, []);
		clQueue.finish();
		clTime += Date.now() - start;
	}
	catch(e) {
		console.innerHTML = e;
	}
}

ScalarField.prototype.diffusion = function() {
	var localWS = [];
	var globalWS = [localThreads, localThreads, localThreads];
	var bufSize = 4 * numCells;

	try {
		scalarCopyKernel.setKernelArg(0, scalarBuffer);
		scalarCopyKernel.setKernelArg(1, scalarTempBuffer);
		scalarCopyKernel.setKernelArg(2, this.dim, WebCL.types.UINT);
		
		var start = Date.now();
		clQueue.enqueueNDRangeKernel(scalarCopyKernel, globalWS.length, [], globalWS, localWS, []);
		clQueue.finish();
		clTime += Date.now() - start;
		
		for(var i = 0; i < 20; i++) {
			scalarDiffusionKernel.setKernelArg(0, scalarBuffer);
			scalarDiffusionKernel.setKernelArg(1, scalarTempBuffer);
			scalarDiffusionKernel.setKernelArg(2, this.dim, WebCL.types.UINT);
			scalarDiffusionKernel.setKernelArg(3, this.dt, WebCL.types.FLOAT);
			scalarDiffusionKernel.setKernelArg(4, this.viscosity, WebCL.types.FLOAT);
			
			var start = Date.now();
			clQueue.enqueueNDRangeKernel(scalarDiffusionKernel, globalWS.length, [], globalWS, localWS, []);
			clQueue.finish();
			clTime += Date.now() - start;
		}
	}
	catch(e) {
		console.innerHTML = e;
	}
	
	this.setBoundaryDensities();
}

ScalarField.prototype.advection = function() {
	var localWS = [];
	var globalWS = [localThreads, localThreads, localThreads];
	var bufSize = 4 * numCells;

	try {
		scalarCopyKernel.setKernelArg(0, scalarBuffer);
		scalarCopyKernel.setKernelArg(1, scalarTempBuffer);
		scalarCopyKernel.setKernelArg(2, this.dim, WebCL.types.UINT);
		
		var start = Date.now();
		clQueue.enqueueNDRangeKernel(scalarCopyKernel, globalWS.length, [], globalWS, localWS, []);
		clQueue.finish();
		clTime += Date.now() - start;
		
		scalarAdvectionKernel.setKernelArg(0, scalarBuffer);
		scalarAdvectionKernel.setKernelArg(1, scalarTempBuffer);
		scalarAdvectionKernel.setKernelArg(2, vectorBuffer);
		scalarAdvectionKernel.setKernelArg(3, this.dim, WebCL.types.UINT);
		scalarAdvectionKernel.setKernelArg(4, this.dt, WebCL.types.FLOAT);
		
		var start = Date.now();	
		clQueue.enqueueNDRangeKernel(scalarAdvectionKernel, globalWS.length, [], globalWS, localWS, []);
		clQueue.finish();
		clTime += Date.now() - start;
	}
	catch(e) {
		console.innerHTML = e;
	}
	
	this.setBoundaryDensities();
}

ScalarField.prototype.setBoundaryDensities = function() {
	var localWS = [];
	var globalWS = [localThreads, localThreads];
	var bufSize = 4 * numCells * 3;
	
	try {
		scalarBoundariesKernel.setKernelArg(0, scalarBuffer);
		scalarBoundariesKernel.setKernelArg(1, this.dim, WebCL.types.UINT);
		
		var start = Date.now();
		clQueue.enqueueNDRangeKernel(scalarBoundariesKernel, globalWS.length, [], globalWS, localWS, []);
		clQueue.finish();
		clTime += Date.now() - start;
	}
	catch(e) {
		console.innerHTML = e;
	}
}

ScalarField.prototype.setCornerDensities = function() {
	this.field[index(0,0,0,dim)] = (this.field[index(1,0,0,dim)] + this.field[index(0,1,0,dim)] + this.field[index(0,0,1,dim)]) / 3;
	this.field[index(0,this.dim+1,0,dim)] = (this.field[index(1,this.dim+1,0,dim)] + this.field[index(0,this.dim,0,dim)] + this.field[index(0,this.dim+1,1,dim)]) / 3;
	this.field[index(this.dim+1,0,0,dim)] = (this.field[index(this.dim,0,0,dim)] + this.field[index(this.dim,1,0,dim)] + this.field[index(this.dim+1,0,1,dim)]) / 3;
	this.field[index(this.dim+1,this.dim+1,0,dim)] = (this.field[index(this.dim,this.dim+1,0,dim)] + this.field[index(this.dim+1,this.dim,0,dim)] + this.field[index(this.dim+1,this.dim+1,1,dim)]) / 3;
	this.field[index(0,0,this.dim+1,dim)] = (this.field[index(1,0,this.dim+1,dim)] + this.field[index(0,1,this.dim+1,dim)] + this.field[index(0,0,this.dim,dim)]) / 3;
	this.field[index(0,this.dim+1,this.dim+1,dim)] = (this.field[index(1,this.dim+1,this.dim+1,dim)] + this.field[index(0,this.dim,this.dim+1,dim)] + this.field[index(0,this.dim+1,this.dim,dim)]) / 3;
	this.field[index(this.dim+1,0,this.dim+1,dim)] = (this.field[index(this.dim,0,this.dim+1,dim)] + this.field[index(this.dim+1,1,this.dim+1,dim)] + this.field[index(this.dim+1,0,this.dim,dim)]) / 3;
	this.field[index(this.dim+1,this.dim+1,this.dim+1,dim)] = (this.field[index(this.dim,this.dim+1,this.dim+1,dim)] + this.field[index(this.dim+1,this.dim,this.dim+1,dim)] + this.field[index(this.dim+1,this.dim+1,this.dim,dim)]) / 3;
}
