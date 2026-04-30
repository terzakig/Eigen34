//
// Analytical polynomial solvers. Implementation by Manolis Lourakis (FORTH)
//
//

#ifndef __POLYSOLVERS_H__
#define __POLYSOLVERS_H__

#include <vector>
#include <cmath>

// #include <iostream>

namespace PolySolvers
{
    // tolerance for floating point inaccuracies
    template <typename P>
    inline constexpr P EPS() {
        return std::numeric_limits<P>::epsilon() * P(100);
    }

    // see http://mathworld.wolfram.com/QuadraticFormula.html
    template <typename P>
    inline std::vector<P> SolveQuadratic(P a, P b, P c)
    {
        static_assert(std::is_floating_point<P>::value, "SolveQuadratic requires a floating point type!");

        P delta = b * b - 4 * a * c;
        P x1, x0;

        if (delta < -EPS<P>())
            return std::vector<P>();
        if (delta < 0)
            delta = 0;
        if (std::abs(a) < EPS<P>())
        {
            // solve first order system
            if (std::abs(b) < EPS<P>())
                return std::vector<P>();

            x0 = -c / b;
            return std::vector<P>({x0});
        }

        if (delta == 0)
        {
            x0 = -b / (2 * a);
            x1 = x0;
            return std::vector<P>({x0});
        }

        P sqrt_delta = std::sqrt(delta);

        // P inv_2a = P(0.5) / a;
        // x0 = (-b + sqrt_delta) * inv_2a;
        // x1 = (-b - sqrt_delta) * inv_2a;

        // more stable computation
        P q = -P(0.5) * (b + std::copysign(sqrt_delta, b));
        // Vieta's formula: x0 * x1 = c / a
        x0 = q / a;
        x1 = c / q;

        return std::vector<P>({x0, x1});
    }

    /* see http://mathworld.wolfram.com/CubicEquation.html */
    template <typename P>
    inline std::vector<P> SolveCubic(P a, P b, P c, P d)
    {
        static_assert(std::is_floating_point<P>::value, "SolveCubic requires a floating point type!");

        P inv_a, b_a, b_a2, c_a, d_a;
        P Q, R, Q3, D, b_a_3;
        P AD, BD;

        P x0, x1, x2;

        if (std::abs(a) < EPS<P>())
        {
            /* solve second order system */
            if (std::abs(b) < EPS<P>())
            {
                /* solve first order system */
                if (std::abs(c) < EPS<P>())
                    return std::vector<P>();

                x0 = -d / c;
                return std::vector<P>({x0});
            }

            return SolveQuadratic<P>(b, c, d);
        }

        /* calculate the normalized form x^3 + a2 * x^2 + a1 * x + a0 = 0 */
        inv_a = P(1) / a;
        b_a = inv_a * b;
        b_a2 = b_a * b_a;
        c_a = inv_a * c;
        d_a = inv_a * d;

        /* solve the cubic equation */
        Q = (3 * c_a - b_a2) / P(9);
        R = (9 * b_a * c_a - 27 * d_a - 2 * b_a * b_a2) / P(54);
        Q3 = Q * Q * Q;
        D = Q3 + R * R;
        b_a_3 = P(1) / P(3) * b_a;

        // check to prevent division by zero and sqrt(negative) later
        if (std::abs(Q) < EPS<P>())
        {
            if (std::abs(R) < EPS<P>())
            {
                x0 = x1 = x2 = -b_a_3;
                return std::vector<P>({x0, x1, x2});
            }
            else
            {
                // use std::cbrt instead of pow
                x0 = std::cbrt(2 * R) - b_a_3;
                return std::vector<P>({x0});
            }
        }

        if (D <= 0)
        {
            /* three real roots */
            // clamp the ratio to [-1.0, 1.0] to prevent acos returning NaN
            P ratio = R / std::sqrt(-Q3);
            ratio = std::max<P>(P(-1), std::min<P>(P(1), ratio));

            P theta = std::acos(ratio);
            P sqrt_Q = std::sqrt(-Q);

            // the three commented lines below give the roots directly
            // x0 = 2 * sqrt_Q * std::cos(theta / 3.0) - b_a_3;
            // x1 = 2 * sqrt_Q * std::cos((theta + 2 * M_PI) / 3.0) - b_a_3;
            // x2 = 2 * sqrt_Q * std::cos((theta + 4 * M_PI) / 3.0) - b_a_3;

            // following code reduces trig calls using the identity cos(a + b) = cos(a)cos(b) - sin(a)sin(b) :
            //   cos(2π/3)=cos(4π/3)=-½  and  sin(2π/3)=-sin(4π/3)=√3/2 (0.866025403784438708), so
            //   cos(φ + 2π/3) = −½ cos(φ) − (√3/2) sin(φ),  cos(φ + 4π/3) = −½ cos(φ) + (√3/2) sin(φ)
            constexpr P half = P(0.5);
            constexpr P sqrt3_over_2 = P(0.86602540378443864676);

            const P phi = theta / P(3);
            P sn_theta3, cs_theta3;
            cs_theta3 = std::cos(phi);
            sn_theta3 = std::sin(phi);
            // can also use sincos(phi, &sn_theta3, &cs_theta3)...

            const P half_c = -half * cs_theta3;
            const P t = sqrt3_over_2 * sn_theta3;

            const P cs_theta3_2pi3 = half_c - t;
            const P cs_theta3_4pi3 = half_c + t;
            const P twice_sqrt_Q = 2 * sqrt_Q;
            x0 = twice_sqrt_Q * cs_theta3 - b_a_3;
            x1 = twice_sqrt_Q * cs_theta3_2pi3 - b_a_3;
            x2 = twice_sqrt_Q * cs_theta3_4pi3 - b_a_3;

            return std::vector<P>({x0, x1, x2});
        }

        /* D > 0, only one real root */

        /* use std::cbrt to handle negative numbers natively */
        const P sqrtD = std::sqrt(D);
        AD = std::cbrt(R + sqrtD);
        BD = std::cbrt(R - sqrtD);

        /* enforce AD * BD = -Q in a numerically stable way */
        if (std::abs(AD) >= std::abs(BD)) {
            if (std::abs(AD) > EPS<P>())
                BD = -Q / AD;
        } else {
            if (std::abs(BD) > EPS<P>())
                AD = -Q / BD;
        }

        /* calculate the sole real root */
        x0 = AD + BD - b_a_3;

        return std::vector<P>({x0});
    }

    /* see http://mathworld.wolfram.com/QuarticEquation.html */
    template <typename P>
    inline std::vector<P> SolveQuartic(P a, P b, P c, P d, P e)
    {
        static_assert(std::is_floating_point<P>::value, "SolveQuartic requires a floating point type!");

        // shortcut to cubic if coefficient is zero...
        if (std::abs(a) < EPS<P>())
        {
            // 0 is a guaranteed root. Factor out x and solve the remaining cubic.
            auto cubic_roots = SolveCubic<P>(a, b, c, d);
            cubic_roots.push_back(0);
            return cubic_roots;
        }

        P inv_a, b2, bc, b3, b_4;
        P r0;
        int nb_real_roots;
        P R2, R_2, R;
        P D2, E2;

        P x0 = 0, x1 = 0, x2 = 0, x3 = 0;

        if (std::abs(a) < EPS<P>())
        {
            x3 = 0;
            return SolveCubic<P>(b, c, d, e);
        }

        /* normalize coefficients */
        inv_a = P(1) / a;
        b *= inv_a;
        c *= inv_a;
        d *= inv_a;
        e *= inv_a;
        b2 = b * b;
        bc = b * c;
        b3 = b2 * b;

        /* solve resultant cubic */
        auto solution3 = SolveCubic<P>(1, -c, d * b - 4 * e, 4 * c * e - d * d - b2 * e);

        if (solution3.size() == 0)
        {
            return std::vector<P>();
        }

        r0 = solution3[0];
        /* calculate R^2 */
        R2 = P(0.25) * b2 - c + r0;
        if (R2 < -EPS<P>())
            return std::vector<P>();

        R = std::sqrt(R2 < 0 ? 0 : R2); // clamp to 0 to prevent NaN

        nb_real_roots = 0;

        /* calculate D^2 and E^2 */
        if (R < EPS<P>())
        {
            P temp = r0 * r0 - 4 * e;
            if (temp < 0)
            {
                D2 = E2 = -1;
            }
            else
            {
                P sqrt_temp = std::sqrt(temp);
                D2 = P(0.75) * b2 - 2 * c + 2 * sqrt_temp;
                E2 = D2 - 4 * sqrt_temp;
            }
        }
        else
        {
            P inv_R = P(1) / R;
            P u = P(0.75) * b2 - 2 * c - R2, v = P(0.25) * inv_R * (4 * bc - 8 * d - b3);
            D2 = u + v;
            E2 = u - v;
        }

        b_4 = P(0.25) * b;
        R_2 = P(0.5) * R;
        if (D2 >= -EPS<P>())
        {
            P D = std::sqrt(D2 < 0 ? 0 : D2); // clamp to 0 if necessary
            P D_2 = P(0.5) * D;
            nb_real_roots = 2;
            x0 = R_2 + D_2 - b_4;
            x1 = x0 - D;
        }

        /* calculate E^2 */
        if (E2 >= -EPS<P>())
        {
            P E = std::sqrt(E2 < 0 ? 0 : E2); // clamp to 0 if necessary
            P E_2 = P(0.5) * E;
            if (nb_real_roots == 0)
            {
                x0 = -R_2 + E_2 - b_4;
                x1 = x0 - E;
                nb_real_roots = 2;
            }
            else
            {
                x2 = -R_2 + E_2 - b_4;
                x3 = x2 - E;
                nb_real_roots = 4;
            }
        }
        switch (nb_real_roots)
        {
        // case 0:
        //     break; // covered by the "default" case
        case 1:
            return std::vector<P>({x0});
            // break;
        case 2:
            return std::vector<P>({x0, x1});
            // break;
        case 3:
            return std::vector<P>({x0, x1, x2});
            // break;
        case 4:
            return std::vector<P>({x0, x1, x2, x3});
            // break;
        default:
            return std::vector<P>(); // just to shut the compiler up....
                                     // break;
        }
    }

    ////// Root polishing with Newton's method

    // polish each root of a cubic polynomial
    template <typename P>
    void PolishCubicRoots(P a, P b, P c, P d, std::vector<P>& roots)
    {
        constexpr int MAX_ITER = 2;
        constexpr P POLISH_TOL = P(4) * PolySolvers::EPS<P>(); // ~4 ULP convergence

        for (auto& r : roots)
        {
            for (int i = 0; i < MAX_ITER; ++i)
            {
                P fx  = ((a * r + b) * r + c) * r + d; // f(x)
                P dfx = (P(3) * a * r + P(2) * b) * r + c; // f'(x)

                if (std::abs(dfx) <= PolySolvers::EPS<P>() * (P(1) + std::abs(r)))
                    break;  // near critical point

                P step = fx / dfx;
                r -= step;

                if (std::abs(step) <= POLISH_TOL * (P(1) + std::abs(r)))
                    break;
            }
        }
    }

    // polish each root of a quartic polynomial
    template <typename P>
    void PolishQuarticRoots(P a, P b, P c, P d, P e, std::vector<P>& roots)
    {
        constexpr int MAX_ITER = 3;  // quartic may benefit from an extra step
        constexpr P POLISH_TOL = P(4) * PolySolvers::EPS<P>(); // ~4 ULP convergence

        for (auto& r : roots)
        {
            for (int i = 0; i < MAX_ITER; ++i)
            {
                P fx = (((a * r + b) * r + c) * r + d) * r + e; // f(x)
                P dfx = ((P(4) * a * r + P(3) * b) * r + P(2) * c) * r + d; // f'(x)

                if (std::abs(dfx) <= PolySolvers::EPS<P>() * (P(1) + std::abs(r)))
                    break;

                P step = fx / dfx;
                r -= step;

                if (std::abs(step) <= POLISH_TOL * (P(1) + std::abs(r)))
                    break;
            }
        }
    }

} // end namespace PolySolvers

#endif
