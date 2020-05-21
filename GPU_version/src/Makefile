objects = ray.o vec.o d_mat_types.o

all: $(objects)
	nvcc  $(objects) -o ray

# Need to use "-x c++" so symbols are name-mangled correctly
# Can also compile with g++
#     g++ -O3 -I. $^ -c -o $@

%.o: %.cu
	nvcc -x cu -O3 --resource-usage -I. -dc  $^ -c -o $@



