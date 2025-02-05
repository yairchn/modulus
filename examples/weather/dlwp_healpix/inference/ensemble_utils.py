
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

def bred_vector_centered(
    ocean_input,
    ocean_model,
    atmos_model,
    atmos_input,
    atmos_coupler,
    ocean_coupler,
    atmos_data_module,
    ocean_data_module,
    ocean_loader_iter,
    atmos_constants,
    ocean_constants,
    noise_dim,  # Must be even! (The original bred_vector expects this and returns size noise_dim+1)
    noise_var,
    device,
    integration_steps=40,
    inflate=False
):
    """
    Produce a centered bred vector ensemble that fits the preallocated size.
    
    The original bred_vector function is called with the given noise_dim and returns an ensemble
    of shape [noise_dim+1, ...], where index 0 is the control and indices 1...noise_dim are perturbed states.
    
    For a centered ensemble we want a control plus equal numbers of positive and negative perturbations:
      final ensemble size = 2N+1.
    To match the preallocated size (noise_dim+1), we require:
          2N+1 = noise_dim+1   -->   N = noise_dim/2.
    
    Thus, when using the centered routine, noise_dim must be even.
    
    This function does the following:
      1. It calls the original bred_vector with noise_dim (unchanged) so that its output is [noise_dim+1, ...].
      2. It computes N = noise_dim//2.
      3. It uses only the first N perturbed members (i.e. indices 1 through 1+N) to form:
             positive branch: control + (perturbation)
             negative branch: control - (perturbation)
      4. It returns the final centered ensemble of shape [2N+1, ...] = [noise_dim+1, ...].
    
    Args:
        noise_dim (int): Must be even. (e.g., if noise_dim=2, then the final ensemble size is 3.)
        noise_var (float): Variance used for the initial bred vector perturbation.
        (Other arguments are passed directly to bred_vector.)
    
    Raises:
        ValueError: If noise_dim is not even.
    
    Returns:
        torch.Tensor: A centered ensemble of shape [noise_dim+1, *state_shape] with:
                      - First N members: v0 + v_bv_i (positive branch)
                      - Middle member:   v0 (control)
                      - Last N members:  v0 - v_bv_i (negative branch)
    """
    # Ensure that noise_dim is even so that the final ensemble size (noise_dim+1) is odd.
    if noise_dim % 2 != 0:
        raise ValueError(f"For centered bred vector, noise_dim must be even, but got noise_dim={noise_dim}.")

    # Compute N such that the final ensemble size will be 2N+1 = noise_dim+1.
    N = noise_dim // 2

    # Call the original bred_vector with the full noise_dim.
    # (This is necessary because your coupler and preallocated arrays expect noise_dim+1 members.)
    final_ens = bred_vector(
        ocean_input,
        ocean_model,
        atmos_model,
        atmos_input,
        atmos_coupler,
        ocean_coupler,
        atmos_data_module,
        ocean_data_module,
        ocean_loader_iter,
        atmos_constants,
        ocean_constants,
        noise_dim,  # Use the original noise_dim (e.g., 2, which makes 3 members)
        noise_var,
        device,
        integration_steps=integration_steps,
        inflate=inflate
    )
    # final_ens now has shape [noise_dim+1, ...]:
    #   - final_ens[0] is the control state v0.
    #   - final_ens[1] ... final_ens[noise_dim] are the perturbed states.

    # Extract the control.
    control = final_ens[0:1]  # shape [1, *state_shape]

    # Use only the first N perturbed states (i.e. indices 1 to 1+N) for centering.
    pos_perturbations = final_ens[1:1+N] - control  # shape [N, *state_shape]

    # Construct the negative branch (mirror the positive deviations).
    neg_ens = control - pos_perturbations  # shape [N, *state_shape]

    # Concatenate to form the centered ensemble:
    #   [v0 + pos, v0, v0 - pos]
    # This yields a final ensemble of shape [N + 1 + N] = [2N+1, ...] = [noise_dim+1, ...].
    big_ens = torch.cat([
        control + pos_perturbations,  # Positive branch (N members)
        control,                      # Control (1 member)
        neg_ens                       # Negative branch (N members)
    ], dim=0)

    return big_ens
