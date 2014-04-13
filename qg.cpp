#include <omp.h>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <random>
#include <vector>
#include <iostream>
using std::cout;
using std::endl;
using std::setw;

const int POPULATION = 75;
const int MUTATION = std::ceil(POPULATION / 100.0f * 3.0f);
const int MAX_ITER = 1E+7;

typedef std::vector<int> Qvec;
typedef std::vector<int>::size_type Qtype;
typedef std::mt19937 RNGtype;
typedef std::uniform_int_distribution<RNGtype::result_type> DStype;

void printq(Qvec qs, int wd = 1) {
    for (auto i: qs) cout << " " << setw(wd) << i;
    cout << "." << endl;
}

inline bool isDublicate(Qtype x, Qvec qs) {
    auto dub = 0;
    for (auto v: qs) 
        if ((x == v) && (++dub == 2)) return true;
    return false;
}

Qvec crossover(Qvec qp1, Qvec qp2, RNGtype engine) {
    auto sz = qp1.size();
    DStype uds_sz(0,sz-1);
    if (sz != qp2.size()) throw std::runtime_error("Lengths are not equal.");
    Qvec qs(sz);
    std::fill(qs.begin(),qs.end(),-1);
    for (auto i = 0; i < sz; i++)
        if (qp1[i] == qp2[i]) qs[i] = qp1[i];
    for (auto i = 0; i < sz; i++)
        if (qs[i] == -1) {
            qs[i] =  uds_sz(engine);
            while (isDublicate(qs[i], qs)) qs[i] = uds_sz(engine);
        }
    return qs; 
}

Qvec gemmation(Qvec qp, RNGtype engine) {
    DStype uds_sz(0, qp.size()-1);
    Qvec qs(qp);
        auto x = uds_sz(engine);
        auto y = uds_sz(engine);
        if (x != y) std::swap(qs[x],qs[y]);
        else { 
            // Do not need clones in population
            std::iota(qs.begin(),qs.end(),static_cast<Qtype>(0));       
            std::shuffle(qs.begin(),qs.end(),engine);
        }
    return qs; 
}

Qtype hits (Qvec qs) {
    Qtype hts = 0;
    Qtype sz = qs.size();
    for (auto i = 0; i < sz - 1; i++) 
        for (auto j = i + 1; j < sz; j++) 
            if (j - i == abs(qs[j] - qs[i])) hts++;
    return hts;
}   

inline bool test(Qvec qs) {
     return (hits(qs)) ? false: true;
}

int main(int argc, char * argv[]) {
    const int sz = (argc > 1) ? std::atoi(argv[1]) : 8;
    const int num_threads = (argc > 2) ? std::atoi(argv[2]) : 1;

    std::vector<Qvec> qs(POPULATION, Qvec(sz));  
    Qvec fit(POPULATION);
    Qvec inds(fit.size());
    std::iota(inds.begin(),inds.end(),static_cast<Qtype>(0)); 
    int num_gen = 1;
    bool debug = true;
 
    DStype uds;
    DStype uds_sz(0,sz-1);
    DStype uds100(0,100);
    DStype uds1(0,1);
    RNGtype engines[num_threads];
    std::random_device rd;
    engines[0].seed(rd());
    for (auto t = 1; t < num_threads; t++)
        engines[t].seed(uds(engines[0]));     

    omp_set_num_threads(num_threads);

    double tm = omp_get_wtime();    
    // First generation
    #pragma omp parallel for
    for (auto p = 0; p < POPULATION; p++){
        auto t = omp_get_thread_num();
        std::iota(qs[p].begin(),qs[p].end(),static_cast<Qtype>(0));       
        std::shuffle(qs[p].begin(),qs[p].end(),engines[t]);
        fit[p] = hits(qs[p]);
    }
    // Main cycle
    while (num_gen++ < MAX_ITER) {
        // Quantile
        std::sort(inds.begin(), inds.end(), 
         [&](Qtype x, Qtype y) {return fit[x] < fit[y];});
        // Check for solution
        if (!fit[inds[0]]) break;
        if (debug && num_gen % 1000 == 0) cout << "The best sol on #" << num_gen <<
         " iteration has " << fit[inds[0]] << " conflict(s)." << endl;
        // New generation: replace the worst solutions (fit > median) by children
        // of the best solutions than children may mutate
        #pragma omp parallel for
        for (auto p = 0; p < POPULATION / 2; p++){
            auto t = omp_get_thread_num();
            auto par1 = inds[p];
            auto par2 = inds[p+1];
            auto chld = inds[POPULATION - p - 1];
            //qs[chld] = crossover(qs[par1],qs[par2],engines[t]);
            if (uds1(engines[t])) qs[chld] = gemmation(qs[par1],engines[t]);
            // mutation
            if (uds100(engines[t]) <= MUTATION) {
                auto x = uds_sz(engines[t]);
                auto y = uds_sz(engines[t]);
                if (x != y) std::swap(qs[chld][x],qs[chld][y]);
            }
        }   
        // eval children
        #pragma omp parallel for
        for (auto p = POPULATION / 2; p < POPULATION; p++) 
            fit[inds[p]] = hits(qs[inds[p]]);
    }        
    if (num_gen == MAX_ITER) {
        cout << sz << "-queen puzzle is NOT solved." << endl;
        return 1;
    }
    if (test(qs[inds[0]])) {
        cout << sz << "-queen puzzle is SOLVED on " << num_gen <<
         "th iteration by " << num_threads << " threads. Time is " << 
         std::fixed << omp_get_wtime() - tm << "s." << endl;
        cout << "solution:";
        printq(qs[inds[0]]);
    }
    return 0;
}
