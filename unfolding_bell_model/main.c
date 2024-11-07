#include <stdio.h>
#include <math.h>
#include <stdlib.h>

float unfold_const_calc()
{
    /*force in nN, distances in nm, attempt frequency in s^-1, energy barrier in kbT*/
    float attempt_freq, energy_barrier, force, transition_dx, k_bT, numerator, denom, exponent, rate_constant;
    attempt_freq = 0.5;
    energy_barrier = 5;
    force = 20;
    k_bT = 1;
    numerator = -1*(energy_barrier - force * transition_dx);
    denom = k_bT;
    exponent = numerator / denom;
    rate_constant = attempt_freq * expf(exponent);
    return rate_constant;
}


float ex_func()
{
    float force = 0.01;
    float rate_constant = unfold_const_calc();
    float extension = force / rate_constant;
    return extension;

}

int main()
{
    /*force in nN, distances in nm, attempt frequency in s^-1, energy in kbT*/
    float transition_dx, extension;
    transition_dx = 0;
    for (transition_dx = 0; transition_dx < 10; transition_dx+=0.5){
        extension = ex_func();
        printf("%5f\n", extension);
    }
    return 0;
}
