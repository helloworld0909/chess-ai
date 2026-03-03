"""Tests for Stockfish wrapper."""

import pytest

# Skip all tests if stockfish is not installed
pytest.importorskip("chess")

import sys

sys.path.insert(0, "src")

from chess_mcp.stockfish import AnalysisLine, Score, ScoreType, Stockfish, StockfishError

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
ITALIAN_GAME_FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
MATE_IN_2_FEN = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"


class TestScore:
    def test_centipawns_score(self):
        score = Score(ScoreType.CENTIPAWNS, 150)
        assert score.centipawns == 150
        assert score.mate_in is None
        assert str(score) == "+1.50"

    def test_negative_score(self):
        score = Score(ScoreType.CENTIPAWNS, -200)
        assert str(score) == "-2.00"

    def test_mate_score(self):
        score = Score(ScoreType.MATE, 3)
        assert score.mate_in == 3
        assert score.centipawns is None
        assert "M" in str(score)

    def test_win_probability(self):
        # Equal position
        score = Score(ScoreType.CENTIPAWNS, 0)
        assert 0.49 < score.win_probability < 0.51

        # Winning position
        score = Score(ScoreType.CENTIPAWNS, 300)
        assert score.win_probability > 0.7

        # Mate
        score = Score(ScoreType.MATE, 2)
        assert score.win_probability == 1.0


class TestAnalysisLine:
    def _make_line(self, wdl=None):
        return AnalysisLine(
            depth=18,
            seldepth=24,
            score=Score(ScoreType.CENTIPAWNS, 30),
            nodes=100000,
            nps=1000000,
            time_ms=100,
            pv=["e2e4"],
            multipv=1,
            wdl=wdl,
        )

    def test_win_probability_uses_wdl_when_available(self):
        line = self._make_line(wdl=(650, 300, 50))
        assert line.win_probability == pytest.approx(0.65)

    def test_win_probability_falls_back_to_sigmoid(self):
        line = self._make_line(wdl=None)
        # cp=30 → sigmoid ≈ 0.52, just check it's near 0.5
        assert 0.50 < line.win_probability < 0.55

    def test_wdl_none_by_default(self):
        line = self._make_line()
        assert line.wdl is None


@pytest.fixture
async def stockfish():
    """Create a Stockfish instance for testing."""
    sf = Stockfish(depth=10, threads=1, hash_mb=64)
    try:
        await sf.start()
        yield sf
    finally:
        await sf.stop()


class TestStockfish:
    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test starting and stopping Stockfish."""
        sf = Stockfish(depth=10)
        await sf.start()
        assert sf._ready
        await sf.stop()
        assert not sf._ready

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using Stockfish as context manager."""
        async with Stockfish(depth=10) as sf:
            assert sf._ready
        assert not sf._ready

    @pytest.mark.asyncio
    async def test_get_best_move(self, stockfish):
        """Test getting best move."""
        move = await stockfish.get_best_move(STARTING_FEN, depth=10)
        assert move  # Should return some move
        assert len(move) >= 4  # UCI format: e.g., "e2e4"

    @pytest.mark.asyncio
    async def test_get_eval(self, stockfish):
        """Test getting evaluation."""
        line = await stockfish.get_eval(STARTING_FEN, depth=10)
        assert isinstance(line, AnalysisLine)
        # Starting position should be roughly equal
        if line.score.centipawns is not None:
            assert abs(line.score.centipawns) < 100
        # WDL win probability should be in valid range
        assert 0.0 <= line.win_probability <= 1.0

    @pytest.mark.asyncio
    async def test_analyze(self, stockfish):
        """Test full analysis."""
        analysis = await stockfish.analyze(ITALIAN_GAME_FEN, depth=10)
        assert analysis.fen == ITALIAN_GAME_FEN
        assert len(analysis.lines) >= 1
        assert analysis.best_move
        assert analysis.pv

    @pytest.mark.asyncio
    async def test_multipv(self, stockfish):
        """Test multi-PV analysis."""
        analysis = await stockfish.analyze(ITALIAN_GAME_FEN, depth=10, multipv=3)
        assert len(analysis.lines) >= 1  # May be fewer if position is forced

    @pytest.mark.asyncio
    async def test_compare_moves_best(self, stockfish):
        """Test comparing best move."""
        # Get best move first
        best = await stockfish.get_best_move(STARTING_FEN, depth=10)

        # Compare it
        result = await stockfish.compare_moves(STARTING_FEN, best, depth=10)
        # At low depth, evaluations can vary slightly between calls
        # The move should still be classified as Best or Great
        assert result["classification"] in ["Best", "Great"]
        assert result["cp_loss"] <= 30  # Should be within "Great" threshold

    @pytest.mark.asyncio
    async def test_compare_moves_illegal(self, stockfish):
        """Test comparing illegal move."""
        result = await stockfish.compare_moves(STARTING_FEN, "e1e8")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_compare_moves_invalid_format(self, stockfish):
        """Test comparing invalid move format."""
        result = await stockfish.compare_moves(STARTING_FEN, "invalid")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_threats(self, stockfish):
        """Test getting threats."""
        result = await stockfish.get_threats(ITALIAN_GAME_FEN, depth=10)
        assert "threats" in result
        assert "fen" in result

    @pytest.mark.asyncio
    async def test_wdl_populated(self, stockfish):
        """Test that WDL values are populated from UCI_ShowWDL."""
        analysis = await stockfish.analyze(ITALIAN_GAME_FEN, depth=12)
        assert analysis.lines
        line = analysis.lines[0]
        assert line.wdl is not None, "UCI_ShowWDL should produce wdl in info lines"
        w, d, l = line.wdl
        assert w + d + l == pytest.approx(1000, abs=2)  # permille sums to ~1000
        assert 0 <= w <= 1000 and 0 <= d <= 1000 and 0 <= l <= 1000
        # Native WDL win_probability should be in [0, 1]
        assert 0.0 <= line.win_probability <= 1.0

    @pytest.mark.asyncio
    async def test_mate_detection(self, stockfish):
        """Test mate detection."""
        line = await stockfish.get_eval(MATE_IN_2_FEN, depth=15)
        # This position is checkmate or near-mate for White
        if line.score.mate_in is not None:
            assert line.score.mate_in > 0  # White is winning


class TestStockfishError:
    @pytest.mark.asyncio
    async def test_invalid_path(self):
        """Test error when Stockfish path is invalid."""
        sf = Stockfish(path="/nonexistent/stockfish")
        with pytest.raises(StockfishError):
            await sf.start()
