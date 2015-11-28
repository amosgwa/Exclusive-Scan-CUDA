#include <stdio.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

template<class T>
void printArray(T &input, int size);

// Debugging
template<class T>
void printArray(T &input, int size) {
  cout << "[ ";
  for(int i = 0; i < size; i++ ) {
    if( i != size-1 ) cout << input[i] << ", ";
    else cout << input[i] << " ]" << endl;
  }
}

// Populate an array with random numbers [1,1000].
void populateArray(int *array, int N){
  srand (time(NULL)); // initialize random seed.

  for(int i = 0; i < N; i++){
    array[i] = (int) rand() % 1000 + 1;
  }
}

// Perform exclusive scan with binary operation addition.
void exclusiveScanSerial(int *array, int *result, int N) {
	result[0] = 0;

	for(int i = 0; i < N-1 ; i++) {
		result[i+1] = result[i] + array[i];
	}
}

// Do not forget to initialize arrays with zero for n*blocksize ; not just n.
__global__ void reduce(int *d_array, int *d_result, int N) {
	
	// First reduce the array.
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	int tId = threadIdx.x;

	// Copy the global memory to shared memory.
	extern __shared__ int s_array[];
	s_array[tId] = d_array[index];
	__syncthreads();

	for(unsigned int s = blockDim.x/2 ; s > 0 ; s >>=1 ) {
		if(tId< s) {
			s_array[tId] += s_array[tId+s];
		}
		__syncthreads();
	}

	if(tId == 0) {
		d_result[blockIdx.x] = s_array[0];
	}
	
}

// Exclusive scan on CUDA.
__global__ void exclusiveScanGPU(int *d_array, int *d_result, int N, int *d_aux) {

	extern __shared__ int temp[]; 

	int realIndex = 2 * threadIdx.x + blockDim.x * 2 * blockIdx.x;

  int threadIndex = threadIdx.x;  
  int index = 2 * threadIndex;   

  int offset = 1;

	// Copy from the array to shared memory.
	temp[index] = d_array[realIndex];
	temp[index+1] = d_array[realIndex+1];  

	// Reduce by storing the intermediate values. The last element will be 
	// the sum of n-1 elements.
	for (int d = blockDim.x; d > 0; d = d/2) {   
		__syncthreads();  

		// Regulates the amount of threads operating.
		if (threadIndex < d)  
		{  
			// Swap the numbers
			int current = offset*(index+1)-1;
			int next = offset*(index+2)-1;
			temp[next] += temp[current];  
		} 

		// Increase the offset by multiple of 2.
		offset *= 2; 
	}

	// Only one thread performs this.
	if (threadIndex == 0) { 
		// Store the sum to the auxiliary array.
		if(d_aux) {
			d_aux[blockIdx.x] = temp[N-1];
		}
		// Reset the last element with identity. Only the first thread will do
		// the job.
		temp[N - 1] = 0; 
	} 

	// Down sweep to build scan.
	for (int d = 1; d < blockDim.x*2; d *= 2) {  

		// Reduce the offset by division of 2.
		offset = offset / 2;

		__syncthreads();  

		if (threadIndex < d)                       
		{  
			int current = offset*(index+1)-1;  
			int next = offset*(index+2)-1;

			// Swap
			int tempCurrent = temp[current];  
			temp[current] = temp[next]; 
			temp[next] += tempCurrent;   
		}  
	}  
	
	__syncthreads(); 

	d_result[realIndex] = temp[index]; // write results to device memory  
	d_result[realIndex+1] = temp[index+1];  	
}

// Summing the increment to the result.
__global__ void sum(int *d_incr, int *d_result, int N) {
	int addThis = d_incr[blockIdx.x];
	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	d_result[tid] += addThis;
}

// Calculate clock.
double calcMS(clock_t &begin, clock_t &end) {
	return double(end-begin)/CLOCKS_PER_SEC;
}

// Check the difference between two arrays.
bool diffCheck(int *a, int *b, int N) {
	for(int i = 0; i < N; i++) {
		if(a[i] != b[i]) {
			return 0;
		}
	}
	return 1;
}

main (int argc, char **argv) {
	if (argc !=2) {
    cout << "Invalid command - Usage : command N " << endl;
    return 0;
  } 

  // Threads size
  int threads = 1024;

  int N = atoi(argv[1]); // Size of the array.

  int blocks = N/threads + ((N%threads == 0) ? 0 : 1);

  int size = blocks * threads;

  int *serial_result = (int *) calloc(size, sizeof(int)); // Allocate dynamic memory for result.

  // Host
  int *h_array = (int *) calloc(size, sizeof(int)); // Allocate dynamic memory for array.  
  int *h_result = (int *) calloc(size, sizeof(int)); // Allocate dynamic memory for result.
  int *h_aux = (int *) calloc(size, sizeof(int)); // Allocate dynamic memory for aux.
  int *h_incr = (int *) calloc(size, sizeof(int)); // Allocate dynamic memory for scanned aux.

  // Device  
  int *d_array;
  int *d_result;
  int *d_aux;

  int *d_incr;

  // Allocate memory on GPU.
  cudaMalloc((void **) &d_aux, size*sizeof(int)); // Allocate dynamic memory for auxiliary array summ. 
  cudaMemset(d_aux,0,size*sizeof(int)); 

  cudaMalloc((void **) &d_incr, size*sizeof(int)); // Allocate dynamic memory for auxiliary array incre. 
  cudaMemset(d_incr,0,size*sizeof(int));

  cudaMalloc((void **) &d_array, size*sizeof(int));
  cudaMemset(d_array,0,size*sizeof(int));

  cudaMalloc((void **) &d_result, size*sizeof(int));
  cudaMemset(d_result,0,size*sizeof(int));

  populateArray(h_array, N); // Populate array with random numbers.

  // Copy from host to device.
 	cudaMemcpy(d_array, h_array, size*sizeof(int), cudaMemcpyHostToDevice);

  // Perform Serial.
  clock_t begin = clock();
  exclusiveScanSerial(h_array, serial_result, size);  
  clock_t end = clock();

  float serialDuration = calcMS(begin, end);

  // Perform on CUDA.
  const dim3 blockSize(threads/2, 1, 1);
  const dim3 gridSize(blocks, 1, 1);

	// First scan.
	begin = clock();
  exclusiveScanGPU<<< gridSize, blockSize, threads * sizeof(int) >>>(d_array, d_result, threads, d_aux);
  end = clock();
  float cudaDuration = calcMS(begin, end);
  cudaDeviceSynchronize();
  // Scan the SUM.
  begin = clock();
  exclusiveScanGPU<<<dim3(1,1,1), blockSize, threads * sizeof(int) >>>(d_aux, d_incr, threads, NULL);
  end = clock();
  cudaDuration += calcMS(begin, end);
  cudaDeviceSynchronize();
  // Add to each block.
  begin = clock();
  sum<<<gridSize, dim3(threads,1,1)>>>(d_incr,d_result, N);
  end = clock();
  cudaDuration += calcMS(begin, end);

  // Copy from Device to Host.
 	cudaMemcpy(h_result, d_result, size*sizeof(int), cudaMemcpyDeviceToHost);
 	cudaMemcpy(h_aux, d_aux, size*sizeof(int), cudaMemcpyDeviceToHost);
 	cudaMemcpy(h_incr, d_incr, size*sizeof(int), cudaMemcpyDeviceToHost);

 	// Calculate Relative percentage.
 	double relative = ((cudaDuration-serialDuration) / serialDuration ) * 100;
 	// Only print when input is less than 1024.
 	if(N <= 1024) {
 		cout << "Input : " << endl;
  	printArray(h_array, N);  
  	cout << "\nSerial Result : " << endl;
 		printArray(serial_result, size);
 		cout << "\nCUDA Result : " << endl;
  	printArray(h_result, size);
  }

  string status = (diffCheck(serial_result, h_result,N)) ? "Passed" : "Failed";
  string failMsg = "";
  if(status == "Failed") {
  	failMsg = "\n\tThis might have failed because the input is too large. It will fail when \n\tthe input is larger than 1024*1024 because we are re-scanning \n\tthe SUM which needs to be less than the number of threads (1024). In my current program, it doesn't \n\tre-scan the SUM when its size exceed the number of threads which means we have to implement \n\tanother rescan for the SUM.\n";
  }


  cout << "\n====================================================================================================" << endl;
  cout << "\tThe data are displayed when N <= 1024 for better performance." << endl;
  cout << "\tThread(s) per block : " << threads << endl;
  cout << "\tBlock(s) per kernel : " << blocks << endl;
  cout << "\tDiff check : " << status << endl << "\t" << failMsg << endl;

  if(status == "Passed") {
	  cout << "\tSerial Execution Time : " << serialDuration << endl;
	  cout << "\tCUDA Execution Time : " << cudaDuration << endl;
	  cout << "\tRelative Performance : " << relative << endl << endl;
	  if(relative <= 0) {
	  	cout << "\tCuda is " << (-1)*relative << "% faster than serial implementation. Yay!!! <(^.^)> \t" << endl;
	  } else {
	  	cout << "\tSerial is " << relative << "% faster than CUDA implementation. Try a bigger input? <(-.- <) \t" << endl;
	  }
	}
    cout << "====================================================================================================\n" << endl;

  // Free memories.
  cudaFree(d_array);
  cudaFree(d_result);
  cudaFree(d_aux);
  cudaFree(d_incr);
  delete [] h_array;
  delete [] h_aux;
  delete [] h_incr;
  delete [] h_result;
  delete [] serial_result;
  return 0;
}
