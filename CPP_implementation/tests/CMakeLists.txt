# mlpack test executable.
add_executable(mlpack_test
  activation_functions_test.cpp
  ada_delta_test.cpp
  ada_grad_test.cpp
  adaboost_test.cpp
  adam_test.cpp
  akfn_test.cpp
  aknn_test.cpp
  ann_layer_test.cpp
  arma_extend_test.cpp
  armadillo_svd_test.cpp
  async_learning_test.cpp
  aug_lagrangian_test.cpp
  augmented_rnns_tasks_test.cpp
  bigbatch_sgd_test.cpp
  binarize_test.cpp
  block_krylov_svd_test.cpp
  cf_test.cpp
  cli_binding_test.cpp
  cli_test.cpp
  cmaes_test.cpp
  cne_test.cpp
  convolution_test.cpp
  convolutional_network_test.cpp
  cosine_tree_test.cpp
  cv_test.cpp
  dbscan_test.cpp
  decision_stump_test.cpp
  decision_tree_test.cpp
  det_test.cpp
  distribution_test.cpp
  drusilla_select_test.cpp
  emst_test.cpp
  fastmks_test.cpp
  feedforward_network_test.cpp
  frankwolfe_test.cpp
  function_test.cpp
  gmm_test.cpp
  gradient_clipping_test.cpp
  gradient_descent_test.cpp
  hmm_test.cpp
  hoeffding_tree_test.cpp
  hpt_test.cpp
  hyperplane_test.cpp
  imputation_test.cpp
  init_rules_test.cpp
  katyusha_test.cpp
  iqn_test.cpp
  kernel_pca_test.cpp
  kernel_test.cpp
  kernel_traits_test.cpp
  kfn_test.cpp
  kmeans_test.cpp
  knn_test.cpp
  krann_search_test.cpp
  ksinit_test.cpp
  lars_test.cpp
  lbfgs_test.cpp
  lin_alg_test.cpp
  line_search_test.cpp
  linear_regression_test.cpp
  load_save_test.cpp
  local_coordinate_coding_test.cpp
  log_test.cpp
  logistic_regression_test.cpp
  loss_functions_test.cpp
  lrsdp_test.cpp
  lsh_test.cpp
  math_test.cpp
  matrix_completion_test.cpp
  maximal_inputs_test.cpp
  mean_shift_test.cpp
  metric_test.cpp
  mlpack_test.cpp
  mock_categorical_data.hpp
  momentum_sgd_test.cpp
  nbc_test.cpp
  nca_test.cpp
  nesterov_momentum_sgd_test.cpp
  nmf_test.cpp
  nystroem_method_test.cpp
  octree_test.cpp
  parallel_sgd_test.cpp
  pca_test.cpp
  perceptron_test.cpp
  prefixedoutstream_test.cpp
  proximal_test.cpp
  python_binding_test.cpp
  q_learning_test.cpp
  qdafn_test.cpp
  quic_svd_test.cpp
  radical_test.cpp
  random_forest_test.cpp
  random_test.cpp
  randomized_svd_test.cpp
  range_search_test.cpp
  rectangle_tree_test.cpp
  recurrent_network_test.cpp
  regularized_svd_test.cpp
  rl_components_test.cpp
  rmsprop_test.cpp
  sa_test.cpp
  sarah_test.cpp
  scd_test.cpp
  sdp_primal_dual_test.cpp
  serialization.cpp
  serialization.hpp
  serialization_test.cpp
  sfinae_test.cpp
  sgd_test.cpp
  sgdr_test.cpp
  smorms3_test.cpp
  snapshot_ensembles.cpp
  softmax_regression_test.cpp
  sort_policy_test.cpp
  spalera_sgd_test.cpp
  sparse_autoencoder_test.cpp
  sparse_coding_test.cpp
  spill_tree_test.cpp
  split_data_test.cpp
  svd_batch_test.cpp
  svd_incremental_test.cpp
  svrg_test.cpp
  termination_policy_test.cpp
  test_function_tools.hpp
  test_tools.hpp
  timer_test.cpp
  tree_test.cpp
  tree_traits_test.cpp
  ub_tree_test.cpp
  union_find_test.cpp
  vantage_point_tree_test.cpp
  main_tests/test_helper.hpp
  main_tests/emst_test.cpp
  main_tests/adaboost_test.cpp
  main_tests/approx_kfn_test.cpp
  main_tests/cf_test.cpp
  main_tests/dbscan_test.cpp
  main_tests/det_test.cpp
  main_tests/decision_tree_test.cpp
  main_tests/decision_stump_test.cpp
  main_tests/linear_regression_test.cpp
  main_tests/logistic_regression_test.cpp
  main_tests/lsh_test.cpp
  main_tests/mean_shift_test.cpp
  main_tests/nbc_test.cpp
  main_tests/nca_test.cpp
  main_tests/nmf_test.cpp
  main_tests/pca_test.cpp
  main_tests/perceptron_test.cpp
  main_tests/preprocess_binarize_test.cpp
  main_tests/preprocess_imputer_test.cpp
  main_tests/preprocess_split_test.cpp
  main_tests/random_forest_test.cpp
  main_tests/softmax_regression_test.cpp
  main_tests/sparse_coding_test.cpp
  main_tests/kmeans_test.cpp
  main_tests/hoeffding_tree_test.cpp
  main_tests/hmm_viterbi_test.cpp
  main_tests/hmm_train_test.cpp
  main_tests/hmm_loglik_test.cpp
  main_tests/hmm_generate_test.cpp
  main_tests/radical_test.cpp
  main_tests/hmm_test_utils.hpp
)

# Link dependencies of test executable.
target_link_libraries(mlpack_test
  mlpack
  ${BOOST_unit_test_framework_LIBRARY}
  ${COMPILER_SUPPORT_LIBRARIES}
)

set_target_properties(mlpack_test PROPERTIES COTIRE_CXX_PREFIX_HEADER_INIT "../core.hpp")
cotire(mlpack_test)

# Copy test data into right place.
add_custom_command(TARGET mlpack_test
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/data/
      ${PROJECT_BINARY_DIR}
)
add_custom_command(TARGET mlpack_test
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E tar xjpf mnist_first250_training_4s_and_9s.tar.bz2
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
)

# The list of long running parallel tests
set(parallel_tests
  "AsyncLearningTest"
  "SdpPrimalDualTest;SVDIncrementalTest;SVDBatchTest;"
  "LocalCoordinateCodingTest;FeedForwardNetworkTest;SparseAutoencoderTest;"
  "GMMTest;CFTest;ConvolutionalNetworkTest;HMMTest;LARSTest;"
  "LogisticRegressionTest")

# Add tests to the testing framework
# Get the list of sources from the test target
get_target_property(test_sources mlpack_test SOURCES)

# Go through the list of test sources and parse the test suite name
foreach(test_file ${test_sources})
  # Regex for parsing files with AUTO_TEST_SUITE
  file(STRINGS ${test_file} test_suite REGEX "BOOST_AUTO_TEST_SUITE\\(.*")
  if(NOT "${test_suite}" STREQUAL "")
    # Get the substring of test_suite within brackets in test_name
    string(REGEX MATCH "\\(.*\\)" test_name ${test_suite})
    # Get the substring excluding the brackets, by calculating the indices
    string(LENGTH ${test_name} end_idx)
    math(EXPR end_idx "${end_idx} - 2")
    string(SUBSTRING ${test_name} "1" ${end_idx} test)
    # Add the test to the testing tool, test is the name of the test suite
    add_test(NAME ${test} COMMAND mlpack_test -t ${test} WORKING_DIRECTORY
      ${CMAKE_BINARY_DIR})
  endif()
  # Regex for parsing files with FIXTURE_TEST_SUITE similarly
  file(STRINGS ${test_file} test_suite REGEX "BOOST_FIXTURE_TEST_SUITE\\(.*,")
  if(NOT "${test_suite}" STREQUAL "")
    # Get the substring of test_suite within brackets and comma in test_name
    string(REGEX MATCH "\\(.*," test_name ${test_suite})
    string(LENGTH ${test_name} end_idx)
    math(EXPR end_idx "${end_idx} - 2")
    string(SUBSTRING ${test_name} "1" ${end_idx} test)
    add_test(NAME ${test} COMMAND mlpack_test -t ${test} WORKING_DIRECTORY
      ${CMAKE_BINARY_DIR})
  endif()
endforeach()

# Use RUN_SERIAL for long running parallel tests
set_tests_properties(${parallel_tests} PROPERTIES RUN_SERIAL TRUE)
