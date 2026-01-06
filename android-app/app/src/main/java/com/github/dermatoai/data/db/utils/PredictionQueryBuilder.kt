package com.github.dermatoai.data.db.utils

import androidx.sqlite.db.SimpleSQLiteQuery
import com.github.dermatoai.domain.entity.PredictionFilter

object PredictionQueryBuilder {

    fun build(filter: PredictionFilter): SimpleSQLiteQuery {
        val sql = StringBuilder("SELECT * FROM prediction_records")
        val args = mutableListOf<Any>()

        var hasWhere = false

        fun appendWhere(condition: String, value: Any) {
            sql.append(if (!hasWhere) " WHERE " else " AND ")
            sql.append(condition)
            args.add(value)
            hasWhere = true
        }

        filter.protocol?.let {
            appendWhere("protocol = ?", it)
        }

        filter.successOnly?.let {
            if (it) appendWhere("is_success = ?", 1)
        }

        filter.label?.let {
            appendWhere("label LIKE ?", "%$it%")
        }

        filter.fromDate?.let {
            appendWhere("created_at >= ?", it)
        }

        filter.toDate?.let {
            appendWhere("created_at <= ?", it)
        }

        sql.append(" ORDER BY created_at DESC")

        return SimpleSQLiteQuery(sql.toString(), args.toTypedArray())
    }
}