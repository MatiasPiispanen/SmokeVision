function VectorField(dim, viscosity, dt, boundaries) {
	this.field = new Float32Array(getFlatSize(dim)*3);
	
	this.dim = dim;
	this.viscosity = viscosity;
	this.dt = dt;
	this.slip = true; // Hardcoded...
	this.vorticityScale = 3.0;
	
	for(var i = 0; i < this.dim+2; i++) {
		for(var j = 0; j < this.dim+2; j++) {
			for(var k = 0; k < this.dim+2; k++) {
				this.field[vindex(i,j,k,0,dim)] = 0.0;
				this.field[vindex(i,j,k,1,dim)] = 0.0;
				this.field[vindex(i,j,k,2,dim)] = 0.0;
			}
		}
	}
}

VectorField.prototype.setTimestep = function(value) {
	this.dt = value;
}

VectorField.prototype.setViscosity = function(value) {
	this.viscosity = value;
}

VectorField.prototype.reset = function() {
	var bufSize = 4 * numCells;
	
	for(var i = 0; i < this.dim+2; i++) {
		for(var j = 0; j < this.dim+2; j++) {
			for(var k = 0; k < this.dim+2; k++) {
				this.field[vindex(i,j,k,0,dim)] = 0.0;
				this.field[vindex(i,j,k,1,dim)] = 0.0;
				this.field[vindex(i,j,k,2,dim)] = 0.0;
			}
		}
	}
	
	clQueue.enqueueWriteBuffer(vectorBuffer, true, 0, bufSize*3, this.getField(), []);
}

VectorField.prototype.getField = function() {
	return this.field;
}

VectorField.prototype.draw = function() {
	
}

VectorField.prototype.step = function(source) {
	this.addField(source);
	this.vorticityConfinement();
	this.diffusion();
	this.projection();
	this.advection();
	this.projection();
}

VectorField.prototype.addField = function(source) {
	vectorAddKernel.setKernelArg(0, vectorBuffer);
	vectorAddKernel.setKernelArg(1, vectorSourceBuffer);
	vectorAddKernel.setKernelArg(2, this.dim, WebCL.types.UINT);
	vectorAddKernel.setKernelArg(3, this.dt, WebCL.types.FLOAT);
	
	try {
		var localWS = [];
		var globalWS = [localThreads, localThreads, localThreads];
		
		var start = Date.now();
		clQueue.enqueueNDRangeKernel(vectorAddKernel, globalWS.length, [], globalWS, localWS, []);
		clQueue.finish();
		clTime += Date.now() - start;
	}
	catch(e) {
		console.innerHTML = e;
	}
}

VectorField.prototype.diffusion = function() {
	var localWS = [];
	var globalWS = [localThreads, localThreads, localThreads];
	var bufSize = 4 * numCells * 3;

	try {
		vectorCopyKernel.setKernelArg(0, vectorBuffer);
		vectorCopyKernel.setKernelArg(1, vectorTempBuffer);
		vectorCopyKernel.setKernelArg(2, this.dim, WebCL.types.UINT);
		
		var start = Date.now();
		clQueue.enqueueNDRangeKernel(vectorCopyKernel, globalWS.length, [], globalWS, localWS, []);
		clQueue.finish();
		clTime += Date.now() - start;

		vectorDiffusionKernel.setKernelArg(0, vectorBuffer);
		vectorDiffusionKernel.setKernelArg(1, vectorTempBuffer);
		vectorDiffusionKernel.setKernelArg(2, this.dim, WebCL.types.UINT);
		vectorDiffusionKernel.setKernelArg(3, this.dt, WebCL.types.FLOAT);
		vectorDiffusionKernel.setKernelArg(4, this.viscosity, WebCL.types.FLOAT);
		
		var start = Date.now();
		clQueue.enqueueNDRangeKernel(vectorDiffusionKernel, globalWS.length, [], globalWS, localWS, []);
		clQueue.finish();
		clTime += Date.now() - start;
	}
	catch(e) {
		console.innerHTML = e;
	}
		
	this.setBoundaryVelocities();
}

VectorField.prototype.projection = function() {
	var localWS = [];
	var globalWS = [localThreads, localThreads, localThreads];
	var bufSize = 4 * numCells * 3;
	try {
		vectorInitFieldKernel.setKernelArg(0, scalarTempBuffer);
		vectorInitFieldKernel.setKernelArg(1, this.dim, WebCL.types.UINT);
	
		var start = Date.now();
		clQueue.enqueueNDRangeKernel(vectorInitFieldKernel, globalWS.length, [], globalWS, localWS, []);
		clQueue.finish();
		clTime += Date.now() - start;
		
		vectorInitFieldKernel.setKernelArg(0, scalarSecondTempBuffer);
		vectorInitFieldKernel.setKernelArg(1, this.dim, WebCL.types.UINT);
		
		var start = Date.now();
		clQueue.enqueueNDRangeKernel(vectorInitFieldKernel, globalWS.length, [], globalWS, localWS, []);
		clQueue.finish();
		clTime += Date.now() - start;
		
		vectorProjectionFirst.setKernelArg(0, vectorBuffer);
		vectorProjectionFirst.setKernelArg(1, scalarTempBuffer);
		vectorProjectionFirst.setKernelArg(2, scalarSecondTempBuffer);
		vectorProjectionFirst.setKernelArg(3, this.dim, WebCL.types.UINT);
		vectorProjectionFirst.setKernelArg(4, 1/this.dim, WebCL.types.FLOAT);
		
		var start = Date.now();
		clQueue.enqueueNDRangeKernel(vectorProjectionFirst, globalWS.length, [], globalWS, localWS, []);
		clQueue.finish();
		clTime += Date.now() - start;
		
		this.setScalarFieldDensities(scalarTempBuffer);
		this.setScalarFieldDensities(scalarSecondTempBuffer);
		
		for(var i = 0; i < 20; i++) {
			vectorProjectionSecond.setKernelArg(0, vectorBuffer);
			vectorProjectionSecond.setKernelArg(1, scalarTempBuffer);
			vectorProjectionSecond.setKernelArg(2, scalarSecondTempBuffer);
			vectorProjectionSecond.setKernelArg(3, this.dim, WebCL.types.UINT);
			vectorProjectionSecond.setKernelArg(4, 1/this.dim, WebCL.types.FLOAT);
			
			var start = Date.now();
			clQueue.enqueueNDRangeKernel(vectorProjectionSecond, globalWS.length, [], globalWS, localWS, []);
			clQueue.finish();
			clTime += Date.now() - start;
			
			this.setScalarFieldDensities(scalarTempBuffer);
		}
		
		vectorProjectionThird.setKernelArg(0, vectorBuffer);
		vectorProjectionThird.setKernelArg(1, scalarTempBuffer);
		vectorProjectionThird.setKernelArg(2, scalarSecondTempBuffer);
		vectorProjectionThird.setKernelArg(3, this.dim, WebCL.types.UINT);
		vectorProjectionThird.setKernelArg(4, 1/this.dim, WebCL.types.FLOAT);
		
		var start = Date.now();
		clQueue.enqueueNDRangeKernel(vectorProjectionThird, globalWS.length, [], globalWS, localWS, []);
		clQueue.finish();
		clTime += Date.now() - start;
		
		this.setBoundaryVelocities();
	}
	catch(e) {
		console.innerHTML = e;
	}
}

VectorField.prototype.advection = function() {
	var localWS = [];
	var globalWS = [localThreads, localThreads, localThreads];
	var bufSize = 4 * numCells * 3;

	try {
		vectorCopyKernel.setKernelArg(0, vectorBuffer);
		vectorCopyKernel.setKernelArg(1, vectorTempBuffer);
		vectorCopyKernel.setKernelArg(2, this.dim, WebCL.types.UINT);
		
		var start = Date.now();
		clQueue.enqueueNDRangeKernel(vectorCopyKernel, globalWS.length, [], globalWS, localWS, []);
		clQueue.finish();
		clTime += Date.now() - start;

		vectorAdvectionKernel.setKernelArg(0, vectorBuffer);
		vectorAdvectionKernel.setKernelArg(1, vectorTempBuffer);
		vectorAdvectionKernel.setKernelArg(2, this.dim, WebCL.types.UINT);
		vectorAdvectionKernel.setKernelArg(3, this.dt, WebCL.types.FLOAT);
			
		var start = Date.now();
		clQueue.enqueueNDRangeKernel(vectorAdvectionKernel, globalWS.length, [], globalWS, localWS, []);
		clQueue.finish();
		clTime += Date.now() - start;
	}
	catch(e) {
		console.innerHTML = e;
	}
		
	this.setBoundaryVelocities();
}


VectorField.prototype.vorticityConfinement = function() {
	var localWS = [];
	var globalWS = [localThreads, localThreads, localThreads];
	var bufSize = 4 * numCells * 3;

	try {
		vectorCopyKernel.setKernelArg(0, vectorBuffer);
		vectorCopyKernel.setKernelArg(1, vectorTempBuffer);
		vectorCopyKernel.setKernelArg(2, this.dim, WebCL.types.UINT);
		
		var start = Date.now();
		clQueue.enqueueNDRangeKernel(vectorCopyKernel, globalWS.length, [], globalWS, localWS, []);
		clQueue.finish();
		clTime += Date.now() - start;

		vectorVorticityFirstKernel.setKernelArg(0, vectorBuffer);
		vectorVorticityFirstKernel.setKernelArg(1, vectorTempBuffer);
		vectorVorticityFirstKernel.setKernelArg(2, scalarTempBuffer);
		vectorVorticityFirstKernel.setKernelArg(3, this.dim, WebCL.types.UINT);
		vectorVorticityFirstKernel.setKernelArg(4, this.dt * this.vorticityScale, WebCL.types.FLOAT);
			
		var start = Date.now();
		clQueue.enqueueNDRangeKernel(vectorVorticityFirstKernel, globalWS.length, [], globalWS, localWS, []);
		clQueue.finish();
		clTime += Date.now() - start;
		
		vectorVorticitySecondKernel.setKernelArg(0, vectorBuffer);
		vectorVorticitySecondKernel.setKernelArg(1, vectorTempBuffer);
		vectorVorticitySecondKernel.setKernelArg(2, scalarTempBuffer);
		vectorVorticitySecondKernel.setKernelArg(3, this.dim, WebCL.types.UINT);
		vectorVorticitySecondKernel.setKernelArg(4, this.dt * this.vorticityScale, WebCL.types.FLOAT);
		
		var start = Date.now();	
		clQueue.enqueueNDRangeKernel(vectorVorticitySecondKernel, globalWS.length, [], globalWS, localWS, []);
		clQueue.finish();
		clTime += Date.now() - start;
	}
	catch(e) {
		console.innerHTML = e;
	}
}

VectorField.prototype.setBoundaryVelocities = function() {
	var localWS = [];
	var globalWS = [localThreads, localThreads];
	var bufSize = 4 * numCells * 3;
	
	try {
		vectorBoundariesKernel.setKernelArg(0, vectorBuffer);
		vectorBoundariesKernel.setKernelArg(1, this.dim, WebCL.types.UINT);
		
		var start = Date.now();
		clQueue.enqueueNDRangeKernel(vectorBoundariesKernel, globalWS.length, [], globalWS, localWS, []);
		clQueue.finish();
		clTime += Date.now() - start;
	}
	catch(e) {
		console.innerHTML = e;
	}
}

VectorField.prototype.setScalarFieldDensities = function(field) {
	var localWS = [];
	var globalWS = [localThreads, localThreads];
	var bufSize = 4 * numCells * 3;
	
	try {
		scalarBoundariesKernel.setKernelArg(0, field);
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
