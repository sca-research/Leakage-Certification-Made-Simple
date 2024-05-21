class mydistMetric {

public:
  mydistMetric() {}

  // The example metric holds no state, so we can mark Evaluate() as static.
  template<typename VecTypeA, typename VecTypeB>
  static double Evaluate(const VecTypeA& a, const VecTypeB& b) {
    
    return arma::norm(a - b, "inf");
  }

};
