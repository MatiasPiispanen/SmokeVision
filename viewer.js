var cubePos = -3;

function Viewer(gl, shaderProgram, scalarField) {
	this.gl = gl;
	this.shaderProgram = shaderProgram;
	
	this.mvMatrix = mat4.create();
    this.pMatrix = mat4.create();
    
    //this.cube = new Cube(gl, shaderProgram, [0.5, 0.5, 0.3, 1.0], gl.CW, true);
	this.scalarField = scalarField;
}

Viewer.prototype.viewport = function() {
	gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
	gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
	mat4.perspective(45.0, gl.viewportWidth / gl.viewportHeight, 0.1, 100.0, this.pMatrix);
	mat4.identity(this.mvMatrix);
}

Viewer.prototype.draw = function () {

	this.scalarField.draw(this);
}

Viewer.prototype.setMatrixUniforms = function() {
	gl.uniformMatrix4fv(this.shaderProgram.pMatrixUniform, false, this.pMatrix);
	gl.uniformMatrix4fv(this.shaderProgram.mvMatrixUniform, false, this.mvMatrix);
}