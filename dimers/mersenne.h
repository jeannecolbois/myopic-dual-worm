#ifndef MERSENNE_H
#define MERSENNE_H
#include <random>
inline std::mt19937_64& random_gen(){
	static std::random_device true_random;
	static std::mt19937_64 mersenne_engine(true_random());
	return mersenne_engine;
}

#endif
