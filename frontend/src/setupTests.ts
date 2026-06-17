import "@testing-library/jest-dom";

// jsdom does not implement Element.prototype.scrollIntoView (no real layout).
// TraceTimeline calls it on its sentinel ref while streaming; stub it so the
// effect doesn't throw under test. This is a no-op polyfill, not production code.
if (typeof Element !== "undefined" && !Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = function scrollIntoView() {};
}
