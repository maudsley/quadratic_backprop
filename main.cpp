#include <iostream>
#include <memory>
#include <map>
#include <vector>
#include <set>
#include <random>

struct graph_node {
  virtual ~graph_node() {}
};

struct var_node : public graph_node {
  double value;
};

struct unary_node : public graph_node {
  virtual double eval(const double x) const = 0;
  virtual double delta(const double x) const = 0;
  std::shared_ptr<graph_node> input;
};

struct binary_node : public graph_node {
  virtual double eval(const double x, const double y) const = 0;
  virtual double delta(const double x, const double dx, const double y, const double dy) const = 0;
  std::shared_ptr<graph_node> lhs;
  std::shared_ptr<graph_node> rhs;
};

struct negate_node : public unary_node {
  virtual double eval(const double x) const {
    return -1 * x;
  }
  virtual double delta(const double x) const {
    return -1;
  }
};

struct mult_node : public binary_node {
  virtual double eval(const double x, const double y) const {
    return x * y;
  }
  virtual double delta(const double x, const double dx, const double y, const double dy) const {
    return dx * y + x * dy; /* d/dx f(x)g(x) = f'(x)g(x) + f(x)g'(x) */
  }
};

struct sum_node : public binary_node {
  virtual double eval(const double x, const double y) const {
    return x + y;
  }
  virtual double delta(const double x, const double dx, const double y, const double dy) const {
    return dx + dy; /* d/dx f(x) + g(x) = f'(x) + g'(x) */
  }
};

class graph_evaluator {
public:
  graph_evaluator(const std::shared_ptr<graph_node>& root) : root_(root) {
  }
  double value() {
    eval(nullptr);
    return values_[root_];
  }
  double delta(const std::shared_ptr<graph_node>& parameter) {
    eval(parameter);
    return deltas_[root_];
  }
private:
  void eval(const std::shared_ptr<graph_node>& parameter) {
    values_.clear();
    deltas_.clear();
    constants_.clear();
    stack_.push_back(root_);
    while (!stack_.empty()) {
      if (std::dynamic_pointer_cast<var_node>(stack_.back())) {
        eval_variable(std::dynamic_pointer_cast<var_node>(stack_.back()), parameter);
      } else if (std::dynamic_pointer_cast<unary_node>(stack_.back())) {
        eval_unary(std::dynamic_pointer_cast<unary_node>(stack_.back()), parameter);
      } else if (std::dynamic_pointer_cast<binary_node>(stack_.back())) {
        eval_binary(std::dynamic_pointer_cast<binary_node>(stack_.back()), parameter);
      }
    }
  }
  void eval_variable(const std::shared_ptr<var_node>& node, const std::shared_ptr<graph_node>& parameter) {
    values_[node] = node->value;
    if (node != parameter) {
      constants_.insert(node); /* hold this variable constant */
      deltas_[node] = 0; /* derivative of a constant is zero */
    } else {
      deltas_[node] = 1; /* d/dx x = 1 */
    }
    stack_.pop_back();
  }
  void eval_unary(const std::shared_ptr<unary_node>& node, const std::shared_ptr<graph_node>& parameter) {
    if (values_.find(node->input) == std::end(values_)) {
      stack_.push_back(node->input); /* need to process unary input */
    } else {
      values_[node] = node->eval(values_[node->input]);
      if (constants_.find(node->input) != std::end(constants_)) {
        constants_.insert(node);
        deltas_[node] = 0;
      } else { /* d/dx f(g(x)) = f'(g(x))g'(x) */
        deltas_[node] = node->delta(values_[node->input]) * deltas_[node->input];
      }
      stack_.pop_back();
    }
  }
  void eval_binary(const std::shared_ptr<binary_node>& node, const std::shared_ptr<graph_node>& parameter) {
    if (values_.find(node->lhs) == std::end(values_)) {
      stack_.push_back(node->lhs); /* need to process lhs input */
    }
    if (values_.find(node->rhs) == std::end(values_)) {
      stack_.push_back(node->rhs); /* need to process rhs input */
    }
    if (stack_.back() == node) { /* have both inputs? evaluate this node */
      values_[node] = node->eval(values_[node->lhs], values_[node->rhs]);
      if (constants_.find(node->lhs) != std::end(constants_) && constants_.find(node->rhs) != std::end(constants_)) { 
        constants_.insert(node);
        deltas_[node] = 0; /* d/dx f(a,b) = 0 */
      } else { /* one or both of the inputs is a parameter */
        deltas_[node] = node->delta(values_[node->lhs], deltas_[node->lhs], values_[node->rhs], deltas_[node->rhs]);
      }
      stack_.pop_back();
    }
  }
  std::shared_ptr<graph_node> root_;
  std::vector<std::shared_ptr<graph_node>> stack_;
  std::set<std::shared_ptr<graph_node>> constants_; /* indicates that a node should be treated as a constant */
  std::map<std::shared_ptr<graph_node>, double> values_; /* output value of each node */
  std::map<std::shared_ptr<graph_node>, double> deltas_; /* output derivative of each node */
};

double target_function(const double x) {
  return 1.23 * x * x + 4.56 * x + 7.89; /* secret numbers (the parameters we will learn) */
}

std::shared_ptr<graph_node> append_error_function(const std::shared_ptr<graph_node>& model, const std::shared_ptr<var_node>& target) {
  /* take the model output, f, and compute the squared distance to the target */
  /* [f(x) - target]^2 */
  std::shared_ptr<negate_node> neg_target = std::make_shared<negate_node>();
  neg_target->input = target;
  std::shared_ptr<sum_node> sum = std::make_shared<sum_node>();
  sum->lhs = model;
  sum->rhs = neg_target;
  std::shared_ptr<mult_node> error = std::make_shared<mult_node>();
  error->lhs = sum;
  error->rhs = sum;
  return error;
}

int main() {
  /* polynomial in x */
  std::shared_ptr<var_node> x = std::make_shared<var_node>();

  /* polynomial coefficients */
  std::shared_ptr<var_node> a = std::make_shared<var_node>();
  std::shared_ptr<var_node> b = std::make_shared<var_node>();
  std::shared_ptr<var_node> c = std::make_shared<var_node>();

  /* quadratic polynomial model */
  std::shared_ptr<mult_node> x_squared = std::make_shared<mult_node>();
  x_squared->lhs = x_squared->rhs = x;

  /* quadratic term */
  std::shared_ptr<mult_node> quad_term = std::make_shared<mult_node>();
  quad_term->lhs = x_squared;
  quad_term->rhs = a;

  /* linear term */
  std::shared_ptr<mult_node> lin_term = std::make_shared<mult_node>();
  lin_term->lhs = x;
  lin_term->rhs = b;

  /* sum of quadratic and linear terms */
  std::shared_ptr<sum_node> quad_lin_sum = std::make_shared<sum_node>();
  quad_lin_sum->lhs = quad_term;
  quad_lin_sum->rhs = lin_term;

  /* final sum include constant term */
  std::shared_ptr<sum_node> f_x = std::make_shared<sum_node>();
  f_x->lhs = quad_lin_sum;
  f_x->rhs = c;

  /* augment the model with the error function */
  std::shared_ptr<var_node> y = std::make_shared<var_node>();
  std::shared_ptr<graph_node> ef_x = append_error_function(f_x, y);

  /* random number generator for generating input samples */
  std::mt19937 rng;

  /* initialize parameters (could be random) */
  a->value = 0.5;
  b->value = 0.5;
  c->value = 0.5;

  /* hyper-parameters */
  const double learning_rate = 0.03;

  /* learning loop */
  for (size_t i = 0; i < 100000; ++i) {
    x->value = rng() / double(rng.max());
    y->value = target_function(x->value);

    graph_evaluator graph(ef_x);

    const double delta_a = graph.delta(a);
    const double delta_b = graph.delta(b);
    const double delta_c = graph.delta(c);

    a->value -= delta_a * learning_rate;
    b->value -= delta_b * learning_rate;
    c->value -= delta_c * learning_rate;
  }

  std::cout << "a = " << a->value << std::endl;
  std::cout << "b = " << b->value << std::endl;
  std::cout << "c = " << c->value << std::endl;

  return 0;
}
