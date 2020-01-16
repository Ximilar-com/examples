/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.classification.tflite;

import android.app.Activity;
import java.io.IOException;
import org.tensorflow.lite.examples.classification.tflite.Classifier.Device;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.ops.NormalizeOp;

/** This TensorFlowLite classifier works with the float MobileNet model. */
public class ClassifierFloatMobileNet extends Classifier {

  /** Ximilar Float MobileNet does not require any additional normalization of the used input.
   * Normalization layer is already present in the model so you don't need to normalize.
   * Setting mean and std as 0.0f and 1.0f, repectively, to bypass the normalization */
  private static final float IMAGE_MEAN = 0.0f;

  private static final float IMAGE_STD = 1.0f;

  /**
   * Float model does not need dequantization in the post-processing. Setting mean and std as 0.0f
   * and 1.0f, repectively, to bypass the normalization.
   */
  private static final float PROBABILITY_MEAN = 0.0f;

  private static final float PROBABILITY_STD = 1.0f;

  /**
   * Initializes a {@code ClassifierFloatMobileNet}.
   *
   * @param activity
   */
  public ClassifierFloatMobileNet(Activity activity, Device device, int numThreads)
      throws IOException {
    super(activity, device, numThreads);
  }

  @Override
  protected String getModelPath() {
    // you can download this file from app.ximilar.com if you have custom pricing plan
    // be aware that most models are with 224x224 resolution, but some of them can have higher
    // if you are not sure then contact tech@ximilar.com
    return "model.tflite";
  }

  /**
   * Be aware that every model has some outputs (named by your labels of your trained task).
   * In the downloaded zip file from app.ximilar.com, file 'labels.txt' should be present.
   * Insert it to the assets folder.
   */
  @Override
  protected String getLabelPath() {
    return "labels.txt";
  }

  @Override
  protected TensorOperator getPreprocessNormalizeOp() {
    return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
  }

  @Override
  protected TensorOperator getPostprocessNormalizeOp() {
    return new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD);
  }
}
