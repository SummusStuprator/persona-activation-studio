"""Backend-aware residual checks shared by scientific and live-generation paths."""

def check(engine,table,controls,pre,post,enabled):
    if engine.model.get('activation_dtype') in ('bfloat16','float16'):
        from .persona_precision import rounding_metrics
        precision=rounding_metrics(table,controls,pre,post,enabled,engine.model['activation_dtype'])
        return precision['max_excess_error'],precision
    from .steering_controls import check_injection
    return check_injection(table,controls,pre,post,enabled),None
