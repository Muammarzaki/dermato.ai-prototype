package com.github.dermatoai.data.db.dao

import androidx.paging.PagingSource
import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.RawQuery
import androidx.sqlite.db.SupportSQLiteQuery
import com.github.dermatoai.data.db.dto.ProtocolStatDto
import com.github.dermatoai.data.db.entity.PredictionRecordEntity
import kotlinx.coroutines.flow.Flow

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

    @Query(
        """
        SELECT 
            protocol, 
            COUNT(*) as total_count,
            SUM(CASE WHEN is_success = 1 THEN 1 ELSE 0 END) as success_count,
            AVG(CASE WHEN is_success = 1 THEN latency_ms ELSE NULL END) as avg_latency_ms
        FROM prediction_records 
        GROUP BY protocol
    """
    )
    fun getProtocolStatistics(): Flow<List<ProtocolStatDto>>

    @Query("SELECT COUNT(id) FROM prediction_records")
    fun getTotalScanCount(): Flow<Int>

    @Query("DELETE FROM prediction_records")
    fun deleteAll()
}