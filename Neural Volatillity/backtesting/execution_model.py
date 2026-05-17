"""
Execution model for the Neural Volatility Surface Forecasting System.

Models bid-ask spread costs, market impact, and effective fill prices for
options trades.  All cost parameters are expressed as fractions of notional.
"""

import logging

logger = logging.getLogger(__name__)

# Typical bid-ask half-widths by option type (as fraction of option price)
_DEFAULT_HALF_WIDTHS: dict[str, float] = {
    "call": 0.01,
    "put": 0.012,
    "straddle": 0.018,
    "strangle": 0.020,
    "spread": 0.015,
}


class ExecutionModel:
    """
    Models realistic transaction costs and fill prices for option trades.

    Parameters
    ----------
    bid_ask_cost_pct : float
        One-way bid-ask cost as a fraction of notional.  Default 0.002 (20 bps).
    market_impact_pct : float
        One-way market impact as a fraction of notional.  Default 0.001 (10 bps).
    """

    def __init__(
        self,
        bid_ask_cost_pct: float = 0.002,
        market_impact_pct: float = 0.001,
    ) -> None:
        if bid_ask_cost_pct < 0:
            raise ValueError("bid_ask_cost_pct must be non-negative.")
        if market_impact_pct < 0:
            raise ValueError("market_impact_pct must be non-negative.")

        self.bid_ask_cost_pct = bid_ask_cost_pct
        self.market_impact_pct = market_impact_pct
        self.total_cost_pct: float = bid_ask_cost_pct + market_impact_pct

        logger.info(
            "ExecutionModel initialised: bid_ask_cost_pct=%.4f market_impact_pct=%.4f "
            "total_cost_pct=%.4f",
            bid_ask_cost_pct,
            market_impact_pct,
            self.total_cost_pct,
        )

    # ------------------------------------------------------------------
    # Transaction cost
    # ------------------------------------------------------------------

    def compute_transaction_cost(
        self,
        notional: float,
        option_type: str = "call",
    ) -> float:
        """
        Compute the total one-way transaction cost for a single option trade.

        Cost = (bid_ask_cost_pct + market_impact_pct) × |notional|

        An additional type-specific bid-ask premium is added based on typical
        market microstructure (wider spreads for complex structures).

        Parameters
        ----------
        notional : float
            Gross dollar notional of the trade.  Sign is ignored.
        option_type : str
            One of {call, put, straddle, strangle, spread}.  Used to apply a
            type-specific bid-ask premium on top of the base cost.

        Returns
        -------
        float
            Transaction cost in dollars (always non-negative).
        """
        abs_notional = abs(notional)
        type_key = option_type.lower()
        extra_half_width = _DEFAULT_HALF_WIDTHS.get(type_key, 0.01)

        base_cost: float = self.total_cost_pct * abs_notional
        type_premium: float = extra_half_width * abs_notional

        total_cost: float = base_cost + type_premium
        logger.debug(
            "compute_transaction_cost: notional=%.2f option_type=%s "
            "base_cost=%.4f type_premium=%.4f total_cost=%.4f",
            abs_notional, option_type, base_cost, type_premium, total_cost,
        )
        return total_cost

    # ------------------------------------------------------------------
    # Net P&L after costs
    # ------------------------------------------------------------------

    def adjust_pnl_for_costs(
        self,
        gross_pnl: float,
        traded_notional: float,
        option_type: str = "call",
    ) -> float:
        """
        Subtract transaction costs from gross P&L to obtain net P&L.

        Parameters
        ----------
        gross_pnl : float
            Pre-cost P&L in dollars.
        traded_notional : float
            Dollar notional traded (used to size the cost deduction).
        option_type : str
            Option type; forwarded to :meth:`compute_transaction_cost`.

        Returns
        -------
        float
            Net P&L after deducting transaction costs.
        """
        cost = self.compute_transaction_cost(traded_notional, option_type)
        net_pnl = gross_pnl - cost
        logger.debug(
            "adjust_pnl_for_costs: gross_pnl=%.4f cost=%.4f net_pnl=%.4f",
            gross_pnl, cost, net_pnl,
        )
        return net_pnl

    # ------------------------------------------------------------------
    # Effective fill price
    # ------------------------------------------------------------------

    def compute_effective_fill(
        self,
        fair_price: float,
        direction: int,
        bid_ask_half: float | None = None,
    ) -> float:
        """
        Compute the effective fill price after crossing the bid-ask spread.

        Buyers pay mid + half-spread; sellers receive mid − half-spread.

        Parameters
        ----------
        fair_price : float
            Mid-market (fair value) price of the option.
        direction : int
            +1 for a buy (we lift the offer), −1 for a sell (we hit the bid).
        bid_ask_half : float or None
            Half the bid-ask spread in dollar terms.  If None, defaults to
            ``bid_ask_cost_pct × fair_price``.

        Returns
        -------
        float
            Effective fill price.

        Raises
        ------
        ValueError
            If direction is not +1 or -1.
        """
        if direction not in (1, -1):
            raise ValueError(f"direction must be +1 or -1; got {direction}.")

        if bid_ask_half is None:
            bid_ask_half = self.bid_ask_cost_pct * fair_price

        effective_fill: float = fair_price + direction * bid_ask_half
        logger.debug(
            "compute_effective_fill: fair_price=%.4f direction=%+d "
            "bid_ask_half=%.4f effective_fill=%.4f",
            fair_price, direction, bid_ask_half, effective_fill,
        )
        return effective_fill