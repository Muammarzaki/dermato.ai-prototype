package com.github.dermatoai.data.db.dao

import androidx.paging.PagingSource
import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.RawQuery
import androidx.sqlite.db.SupportSQLiteQuery
import com.github.dermatoai.data.db.entity.PredictionRecordEntity

@Dao
interface PredictionRecordDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(record: PredictionRecordEntity): Long

    @Query("DELETE FROM prediction_records WHERE id = :id")
    suspend fun deleteById(id: Long)

    @Query("SELECT * FROM prediction_records WHERE id = :id LIMIT 1")
    suspend fun getById(id: Long): PredictionRecordEntity?

    @RawQuery(observedEntities = [PredictionRecordEntity::class])
    fun pagingWithFilter(
        query: SupportSQLiteQuery
    ): PagingSource<Int, PredictionRecordEntity>

}