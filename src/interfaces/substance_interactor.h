/*
 * Copyright 2026 Amina Sokoli (@aminatpwk)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     SPDX-License-Identifier: Apache-2.0
 */

#ifndef SUBSTANCE_INTERACTOR_H_
#define SUBSTANCE_INTERACTOR_H_

#include "core/real_t.h"

namespace bdm {

/// Interface for agents that interact with diffusion substances
/// Any cell type that consumes or secretes substances must implement this
/// interface. This avoids dynamic_cast chains in the diffusion grid's
/// ComputeConsumptionsSecretions.
class ISubstanceInteractor {
 public:
  virtual ~ISubstanceInteractor() = default;

  /// Compute the new concentration of a substance after consumption/secretion.
  ///
  /// @param substance_id  The continuum ID of the diffusing substance
  /// @param old_concentration  Current concentration at the cell's voxel
  /// @return New concentration after the cell's interaction
  virtual real_t ConsumeSecreteSubstance(int substance_id,
                                         real_t old_concentration) = 0;
};

}  // namespace bdm

#endif  // SUBSTANCE_INTERACTOR_H_