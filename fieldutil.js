function getFlatSize(dim) {
	return (dim+2)*(dim+2)*(dim+2);
}

function getWGSize(dim) {
	return (getFlatSize(dim) + 32) / 32;
}

function index(i, j, k, dim) {
	return (i * (dim+2) * (dim+2)) + (j * (dim+2)) + k;
}

function vindex(i, j, k, d, dim) {
	return 3*index(i, j, k, dim) + d;
}
