
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

def inflate_perturbations(ocean_dx: torch.Tensor, noise_amplitude: torch.Tensor) -> torch.Tensor:
    """
    Inflate the ocean perturbations.
    
    Args:
    ocean_dx (torch.Tensor): Ocean perturbations with shape [ensemble_size, *state_shape]
    noise_amplitude (torch.Tensor): Amplitude of the noise used for inflation
    
    Returns:
    torch.Tensor: Inflated ocean perturbations
    """
    mean_perturbation = ocean_dx.mean(dim=0, keepdim=True)
    centered_perturbations = ocean_dx - mean_perturbation
    inflated_perturbations = ocean_dx + noise_amplitude * centered_perturbations
    return inflated_perturbations


def bred_vector(ocean_input, ocean_model, atmos_model, atmos_input, atmos_coupler, ocean_coupler,
                atmos_data_module, ocean_data_module, ocean_loader_iter,
                atmos_constants, ocean_constants,
                noise_dim, noise_var, device, integration_steps=40, inflate=False):
    # Set noise amplitude
    noise_amplitude = torch.tensor(noise_var ** 0.5, device=device)

    # Initialize perturbations for ocean_input[0][1:]
    ocean_dx = noise_amplitude * torch.randn_like(ocean_input[0][1:], device=device)

    # Apply initial perturbations to ensemble members beyond the control run
    ocean_input[0][1:] += ocean_dx

    # Run atmosphere model for the initial step (breeding_cycle == 0)
    with torch.no_grad():
        atmos_output = atmos_model(atmos_input)

    # Use atmos_output to set forcing for ocean model
    ocean_coupler.set_coupled_fields(atmos_output.cpu())

    # Run ocean model with perturbed ocean_input[0]
    with torch.no_grad():
        ocean_output = ocean_model(ocean_input)

    # Update ocean perturbations
    ocean_dx = ocean_output[1:] - ocean_output[0:1]

    # Optionally apply inflation
    if inflate:
        ocean_dx = inflate_perturbations(ocean_dx, noise_amplitude)

    # Update ocean_input[0][1:] for the next breeding cycle
    ocean_input[0][1:] = ocean_input[0][0:1] + ocean_dx

    # Start breeding cycles from the second iteration
    for breeding_cycle in range(1, integration_steps):
        # Atmosphere model
        atmos_coupler.set_coupled_fields(ocean_output.cpu())

        # Generate next atmosphere inputs using next_integration
        atmos_input = [k.to(device) if k is not None else None for k in atmos_data_module.test_dataset.next_integration(
                                                                atmos_output,
                                                                constants=atmos_constants,
                                                            )]

        # Expand atmos_input[1] if necessary
        if noise_dim > 0 and atmos_input[1] is not None:
            atmos_input[1] = atmos_input[1].expand(noise_dim + 1, *atmos_input[1].shape[1:]).clone()

        # Run atmosphere model
        with torch.no_grad():
            atmos_output = atmos_model(atmos_input)

        # Ocean model
        ocean_coupler.set_coupled_fields(atmos_output.cpu())

        if ocean_constants is None:
            ocean_constants = ocean_input[2]
        # Generate next ocean inputs using next_integration
        ocean_input = [k.to(device) if k is not None else None for k in ocean_data_module.test_dataset.next_integration(
                                                                ocean_output,
                                                                constants=ocean_constants,
                                                            )]

        # Expand ocean_input[0], ocean_input[1] if necessary
        if noise_dim > 0:
            if ocean_input[0] is not None:
                ocean_input[0] = ocean_input[0].expand(noise_dim + 1, *ocean_input[0].shape[1:]).clone()
                # Apply the updated perturbations to ocean_input[0][1:]
                ocean_input[0][1:] = ocean_input[0][0:1] + ocean_dx
            if ocean_input[1] is not None:
                ocean_input[1] = ocean_input[1].expand(noise_dim + 1, *ocean_input[1].shape[1:]).clone()

        # Run ocean model with perturbed ocean_input[0]
        with torch.no_grad():
            ocean_output = ocean_model(ocean_input)

        # Update ocean perturbations
        ocean_dx = ocean_output[1:] - ocean_output[0:1]

        # Optionally apply inflation
        if inflate:
            ocean_dx = inflate_perturbations(ocean_dx, noise_amplitude)

        # Update ocean_input[0][1:] for the next breeding cycle
        ocean_input[0][1:] = ocean_input[0][0:1] + ocean_dx

    # Return the updated ocean_input[0]
    # After breeding cycles, apply scaling to perturbations

    # Compute the norm of the control run
    control_norm = torch.norm(ocean_input[0][0])

    # Flatten the tensors for computing norms
    ocean_dx_flat = (ocean_input[0][1:] + ocean_dx).reshape(ocean_input[0][1:].shape[0], -1)

    # Compute norms of each ensemble member's perturbed state
    perturbed_norms = torch.norm(ocean_dx_flat, dim=1)  # Shape: [noise_dim]

    # Compute gamma for each ensemble member
    gammas = control_norm / perturbed_norms  # Shape: [noise_dim]

    # Reshape gammas to match ocean_dx dimensions
    gammas = gammas.view(-1, *([1] * (ocean_dx.dim() - 1)))  # Expands to [noise_dim, 1, 1, 1, ...]

    # Scale the perturbations
    ocean_dx *= gammas

    # Update ocean_input[0][1:] with scaled perturbations
    ocean_input[0][1:] = ocean_input[0][0:1] + ocean_dx

    return ocean_input[0]