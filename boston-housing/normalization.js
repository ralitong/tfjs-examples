/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

/**
 * Calculates the mean and standard deviation of each column of a data array.
 *
 * @param {Tensor2d} data Dataset from which to calculate the mean and
 *                        std of each column independently.
 *
 * @returns {Object} Contains the mean and standard deviation of each vector
 *                   column as 1d tensors.
 */
export function determineMeanAndStddev(data) {
  const dataMean = data.mean(0);
  // TODO(bileschi): Simplify when and if tf.var / tf.std added to the API.
  const diffFromMean = data.sub(dataMean);
  const squaredDiffFromMean = diffFromMean.square();
  const variance = squaredDiffFromMean.mean(0);
  const dataStd = variance.sqrt();
  return { dataMean, dataStd };
}

/**
 * Given expected mean and standard deviation, normalizes a dataset by
 * subtracting the mean and dividing by the standard deviation.
 *
 * @param {Tensor2d} data: Data to normalize. Shape: [batch, numFeatures].
 * @param {Tensor1d} dataMean: Expected mean of the data. Shape [numFeatures].
 * @param {Tensor1d} dataStd: Expected std of the data. Shape [numFeatures]
 *
 * @returns {Tensor2d}: Tensor the same shape as data, but each column
 * normalized to have zero mean and unit standard deviation.
 */
export function normalizeTensor(data, dataMean, dataStd) {
  return data.sub(dataMean).div(dataStd);
}

/**
 * Normalizes a dataset by subtracting the mean and dividing by the standard
 * deviation.
 *
 * @param {Tensor2D} data: Data to normalize. Shape: [batch, numFeatures].
 * @returns {Tensor2D}: Tensor the same shape as data, but each column
 * normalized to have zero mean and unit standard deviation.
 */
export function normalizeTensorWithZScore(data) {
  const { mean, std } = determineMeanAndStddev(data);
  return normalizeTensor(data, mean, std);
}

/**
 * Normalizes a dataset by subtracting the min and dividing by the max.
 *
 * @param {Tensor2d} data: Data to normalize. Shape: [batch, numFeatures].
 *
 * @returns {Tensor2d}: Tensor the same shape as data, but each column
 * normalized to have minimum 0 and maximum 1.
 */
export function normalizeTensorWithMinMax(data) {
  const min = data.min(0);
  const max = data.max(0);
  return data.sub(min).div(max.sub(min));
}

/**
 * Normalizes a dataset by taking the log of each element.
 * @param {Tensor2d} data: Data to normalize. Shape: [batch, numFeatures].
 *
 * @returns {Tensor2d}: Tensor the same shape as data, but each column
 * normalized by taking the log base 10 of each element.
 */
export function normalizeTensorWithLogScaling(data) {
  return data.log().div(tf.log(10));
}

/**
 * Normalizes a dataset by subtracting the median and dividing by the interquartile range.
 * RobustScaling = (data - median) / IQR
 * where IQR = Q3 - Q1
 * @param {Tensor2d} data: Data to normalize. Shape: [batch, numFeatures].
 *
 * @returns {Tensor2d}: Tensor the same shape as data, but each column
 */
export function normalizeTensorWithRobustScaling(data) {
  const median = data.median(0);
  const quartiles = data.quartile(0);
  const min = quartiles.slice([0], [1]); // Lower quartile. Q1
  const max = quartiles.slice([2], [3]); // Upper quartile. Q3
  return data.sub(median).div(max.sub(min));
}
