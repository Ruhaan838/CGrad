# AutoGrad & GradMode
- Some methods in AutoGrad are used in arithmetic but you can use them explicitly for your propose.

## Basics

- ### with AutoGrad.no_grad
    ```python
    AutoGrad.no_grad() -> None
    ```

    - this will stop the caculation of autograd in with in the block of no_grad.

    - Example:
        ```python
        with AutoGrad.no_grad():
            z = w1 @ x1
            print(z.requires_grad)#False
        ```
- ### GradMode:
    - #### GradMode.is_enabled
    ```python
    GradMode.is_enabled() -> bool
    ```
    - Give the current status of grad for Tensor.
    - #### GradMode.set_enabled
    ```python
    GradMode.set_enabled(mode) -> None
    ```
    - Set the mode of requires_grad.