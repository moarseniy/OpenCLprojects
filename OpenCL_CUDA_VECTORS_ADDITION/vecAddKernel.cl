__kernel void addVectors(__global const float *a, __global const float *b, __global float *c) 
{
	int ind = get_global_id(0);
	c[ind] = a[ind] + b[ind];
}
