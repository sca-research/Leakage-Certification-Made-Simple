# Recurse into each method mlpack provides.
set(DIRS
  # mvu # Note: this implementation of MVU does not work.  See #189.
  adaboost
  amf
  ann
  approx_kfn
  block_krylov_svd
  cf
  dbscan
  decision_stump
  decision_tree
  det
  emst
  fastmks
  gmm
  hmm
  hoeffding_trees
  kernel_pca
  kmeans
  lars
  linear_regression
  local_coordinate_coding
  logistic_regression
  lsh
  matrix_completion
  mean_shift
  naive_bayes
  nca
  neighbor_search
  nmf
  nystroem_method
  pca
  perceptron
  preprocess
  quic_svd
  radical
  random_forest
  randomized_svd
  range_search
  rann
  regularized_svd
  reinforcement_learning
  softmax_regression
  sparse_autoencoder
  sparse_coding
  sparse_svm
)

foreach(dir ${DIRS})
    add_subdirectory(${dir})
endforeach()

set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
set(MLPACK_PYXS ${MLPACK_PYXS} PARENT_SCOPE)
