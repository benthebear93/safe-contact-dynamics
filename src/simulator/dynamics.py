def lagrangian_derivatives(model, q, v):
    D1L = -model.C_func(q, v)
    D2L = model.M_func(q) @ v
    return D1L, D2L


def dynamics_auto(model, *args):
    h, q0, q1, u1, w1, Lambda1, q2 = args

    qm1 = (q0 + q1) / 2
    vm1 = (q1 - q0) / h
    qm2 = (q1 + q2) / 2
    vm2 = (q2 - q1) / h

    D1L1, D2L1 = lagrangian_derivatives(model, qm1, vm1)
    D1L2, D2L2 = lagrangian_derivatives(model, qm2, vm2)

    # print(D2L1.shape)
    # print(D2L2.shape)
    # print(D1L1.shape)
    # print((0.5 * h * D1L1).shape)
    # print((model.input_jacobian(qm2) @ u1).shape)
    # print(Lambda1.shape)
    dyn = (
        D2L1
        - D2L2
        + 0.5 * h * D1L1
        + 0.5 * h * D1L2
        + model.input_jacobian(qm2) @ u1
        + Lambda1
    )
    return dyn
