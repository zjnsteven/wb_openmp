#include <Rcpp.h>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <omp.h>


using namespace Rcpp;
// [[Rcpp::export]]
NumericVector splitnc(NumericMatrix y,IntegerVector x){
  std::vector<int> ux;
  for(int i=0;i<x.size();i++){
    //std::cout<<x[i]<<std::endl;
    ux.push_back(x[i]);
  }
  std::sort(ux.begin(),ux.end());
  ux.erase(std::unique(ux.begin(),ux.end()),ux.end());
  int n = ux.size();
  NumericVector out((2*n-1),0.0);
  NumericVector goodness(n-1);
  NumericVector mean(n);
  NumericVector count(n,0.0);
  NumericVector sum(n,0.0);
  int len = y.nrow();
   for(int i=0;i<len;i++){
     for(int j=0;j<n;j++){
       if(x[i] == ux[j]){
         count[j] = count[j] + 1;
         sum[j] = sum[j] + y(i,0);
         break;
       }
     }
   }
   for(int i=0;i<n;i++){
     mean[i] = sum[i] / count[i];
   }
   
//rank X
  for(int i=0;i<n-1;i++){
    for(int j=i+1;j<n;j++){
      if(mean[i]>mean[j]){
        double tmp = mean[j];
        mean[j] = mean[i];
        mean[i] = tmp;
        double temp = ux[j];
        ux[j] = ux[i];
        ux[i] = temp;
      }
    }
    
  }
  #pragma omp parallel for
  for(int j=0;j<n-1;j++){
    double rss = 0.0;
    double wmeanleft = 0.0;
    double wmeanright = 0.0;
    double sumTrtleft = 0.0;
    double sumUntrtleft = 0.0;
    double sumTrtWtleft = 0.0;
    double sumUntrtWtleft = 0.0;
    double sumTrtright = 0.0;
    double sumUntrtright = 0.0;
    double sumTrtWtright = 0.0;
    double sumUntrtWtright = 0.0;
    #pragma omp parallel for
    for(int i=0;i<len;i++){
      int pos = std::find(ux.begin(),ux.end(),x[i]) - ux.begin();
      if(pos<=j){
        if(y(i,1) == 1.0){
          sumTrtleft = sumTrtleft + y(i,0)/y(i,2);
          sumTrtWtleft = sumTrtWtleft + 1/y(i,2);
        }
        else{
          sumUntrtleft = sumUntrtleft + y(i,0)/(1-y(i,2));
          sumUntrtWtleft = sumUntrtWtleft + 1/(1-y(i,2));
        }
      }
      else{
        if(y(i,1) == 1){
          sumTrtright = sumTrtright + y(i,0)/y(i,2);
          sumTrtWtright = sumTrtWtright + 1/y(i,2);
        }
        else{
          sumUntrtright = sumUntrtright + y(i,0)/(1-y(i,2));
          sumUntrtWtright = sumUntrtWtright + 1/(1-y(i,2));
        }
      }
    }
    if(sumTrtWtleft != 0 && sumUntrtWtleft !=0 ){
      wmeanleft =  sumTrtleft/sumTrtWtleft - sumUntrtleft/sumUntrtWtleft;
      for(int i=0;i<len;i++){
        int pos = std::find(ux.begin(),ux.end(),x[i]) - ux.begin();
        if(pos<=j){
          rss = rss + (y(i,3) - wmeanleft)*(y(i,3) - wmeanleft);
        }
      }
    }
    else{
      goodness[j] = 0;
      continue;
    }
    
    if(sumTrtWtright != 0 && sumUntrtWtright !=0 ){
      wmeanright =  -sumTrtright/sumTrtWtright + sumUntrtright/sumUntrtWtright;
      for(int i=0;i<len;i++){
        int pos = std::find(ux.begin(),ux.end(),x[i]) - ux.begin();
        if(pos>j){
          rss = rss + (y(i,3) + wmeanright)*(y(i,3) + wmeanright);
        }
      }
      goodness[j] =  1/rss ;
    }
    else{
      goodness[j] = 0;
    }
  }
    #pragma omp parallel for
    for(int i=0;i<n-1;i++){
      out[i] = goodness[i];
    }
    #pragma omp parallel for
    for(int i=n-1;i<=2*(n-1);i++){
      out[i] = ux[i-n+1];
    }
    return out;

}

// [[Rcpp::export]]
NumericVector node_evaluate(NumericMatrix y){
  double sumTrt = 0.0;
  double sumUntrt = 0.0;
  double sumTrtWt = 0.0;
  double sumUntrtWt = 0.0;
  int nrow = y.nrow();
  double wmean = 0.0;
  double rss = 0.0;
  for(int i=0;i<nrow;i++){
    if(y(i,1) == 1.0){
      sumTrt = sumTrt + y(i,0)/y(i,2);
      sumTrtWt = sumTrtWt + 1/y(i,2);
    }
    else{
      sumUntrt = sumUntrt + y(i,0)/(1-y(i,2));
      sumUntrtWt = sumUntrtWt + 1/(1-y(i,2));
    }
  }
  if(sumTrtWt != 0 && sumUntrtWt !=0 ){
    wmean =  sumTrt/sumTrtWt - sumUntrt/sumUntrtWt;
  }
  for(int i=0;i<nrow;i++){
    rss = rss + (y(i,3) - wmean)* (y(i,3) - wmean);
  }
  NumericVector out(2);
  out[0] = wmean;
  out[1] = rss;
  return out;
}
// [[Rcpp::export]]

  NumericVector splitc(NumericMatrix y){
    int n = y.nrow();
    NumericVector out(2*(n-1),0.0);
    NumericVector goodness(n-1);
    NumericVector direction(n-1);
    int max;
    max=omp_get_max_threads();
    omp_set_num_threads(max);
    #pragma omp parallel for
    for(int j=0;j<n-1;j++){
      double rss = 0.0;
      double wmeanleft = 0.0;
      double wmeanright = 0.0;
      double sumTrtleft = 0.0;
      double sumUntrtleft = 0.0;
      double sumTrtWtleft = 0.0;
      double sumUntrtWtleft = 0.0;
      double sumTrtright = 0.0;
      double sumUntrtright = 0.0;
      double sumTrtWtright = 0.0;
      double sumUntrtWtright = 0.0;
      // left child
      for(int i=0;i<=j;i++) {
        if(y(i,1) == 1.0){
          sumTrtleft = sumTrtleft + y(i,0)/y(i,2);
          sumTrtWtleft = sumTrtWtleft + 1/y(i,2);
        }
        else{
          sumUntrtleft = sumUntrtleft + y(i,0)/(1-y(i,2));
          sumUntrtWtleft = sumUntrtWtleft + 1/(1-y(i,2));
        }
      }
      
      if(sumTrtWtleft != 0 && sumUntrtWtleft !=0 ){
        wmeanleft =  sumTrtleft/sumTrtWtleft - sumUntrtleft/sumUntrtWtleft;
        for(int i=0;i<=j;i++) {
          rss = rss + (y(i,3) - wmeanleft)*(y(i,3) - wmeanleft);
        }
      }
      else{
        goodness[j] = 0;
        continue;
      }
      // right child
      for(int i=j+1;i<n;i++){
        if(y(i,1) == 1){
          sumTrtright = sumTrtright + y(i,0)/y(i,2);
          sumTrtWtright = sumTrtWtright + 1/y(i,2);
        }
        else{
          sumUntrtright = sumUntrtright + y(i,0)/(1-y(i,2));
          sumUntrtWtright = sumUntrtWtright + 1/(1-y(i,2));
        }
      }
      if(sumTrtWtright != 0 && sumUntrtWtright !=0 ){
        wmeanright =  -sumTrtright/sumTrtWtright + sumUntrtright/sumUntrtWtright;
        for(int i=j+1;i<n;i++){
          rss = rss + (y(i,3) + wmeanright)*(y(i,3) + wmeanright);
        }
        goodness[j] =  1/rss ;
        direction[j] = wmeanleft>0 ? 1:-1;
      }
      else{
        goodness[j] = 0;
      }
    }
    for(int i=0;i<n-1;i++){
      out[i] = goodness[i];
    }
    for(int i=n-1;i<2*(n-1);i++){
      out[i] = direction[i-n+1];
    }
    return out;
    
  }
